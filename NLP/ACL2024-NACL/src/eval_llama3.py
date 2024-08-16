import json
from pathlib import Path
import time
from typing import List, Tuple, Any
import unicodedata

import paddle
import paddle.distributed as dist
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from llama_nacl_patch import replace_llama_attn_with_nacl_eviction
from paddlenlp.transformers.llama.modeling import LlamaAttention

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
    ALL_DATA_NAMES,
    DATA_NAME_TO_SAMPLES,
)
from args import parse_args


MAX_POSITION_ID = 130 * 1024  # Determined by the model
TRUNCATE_LEN = 127 * 1024


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    elif manner == "left":
        return input[-max_length:]
    elif manner == "right":
        return input[:max_length]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    input = unicodedata.normalize("NFC", input)
    tokens = tok.tokenizer.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok._decode(tokens, skip_special_tokens=True)

def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    input = tok(input_text, truncation=False, return_tensors="pd")
    input["decode_strategy"] = "greedy_search" # "greedy_search", "sampling" and "beam_search"
    # input["temperature"] = 0.6
    # input["top_p"] = 0.9
    input["max_new_tokens"] = max_tokens
    outputs = model.generate(**input)

    output = outputs[0].tolist()[0]
    output = tok._decode(output, skip_special_tokens=True)
    print("Generation:", output)
    return output

def load_model(model_name):

    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_flash_attention=True,
        use_flash_attention_for_generation=True,
        use_last_token_for_generation=True,
        immediate_clear_past_key_value=True,
        max_position_embeddings=MAX_POSITION_ID,
        dtype="bfloat16"
    )
    print("Time taken:", round(time.time() - start_time))

    return model, tok  # type: ignore

if __name__ == "__main__":
    
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    print(f'Using {dist.get_world_size()} GPUs to evaluation', flush=True)

    if dist.get_world_size() > 1:
        dist.fleet.init(is_collective=True)

    # Model
    model_name = args.model_name
    if args.enable_nacl_evict:
        if not hasattr(LlamaAttention, "kvcache_eviction"):
            print("Replace LLaMA Attention to support NaCL Eviction. Please update paddlenlp to skip this patch!")
            replace_llama_attn_with_nacl_eviction()
    model, tok = load_model(args.model_path)

    if args.enable_nacl_evict:
        kvcache_eviction = {
            "proxy_tokens_ratio": args.proxy_tokens_ratio,
            "sink_tokens": args.sink_tokens,
            "proxy_token_keep_ratio": args.proxy_token_keep_ratio,
            "random_token_keep_ratio": args.random_token_keep_ratio,
            "token_protect_ratio": args.token_protect_ratio,
            "min_eviction_seqlen": args.min_eviction_seqlen
        }
        model.config.kvcache_eviction = kvcache_eviction
        print(f"Enable NACL eviction, config: {kvcache_eviction}")


    # Split Data
    data_names = args.task
    if args.task == "all":
        data_names = ALL_DATA_NAMES
    else:
        data_names = data_names.split(',')

    # Sort for multi gpus balance
    data_names = sorted(data_names, key=lambda x: DATA_NAME_TO_SAMPLES[x])

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    data_names = data_names[rank::world_size]

    # For Loop
    for data_name in data_names:
        # Data
        max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        result_dir = Path(args.output_dir, model_name)
        result_dir.mkdir(exist_ok=True, parents=True)
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.stop_idx is None:
            start_idx = args.start_idx
            stop_idx = len(examples)
            output_path = (
                result_dir / f"preds_{data_name}.jsonl"
            )
        else:
            start_idx = args.start_idx
            stop_idx = args.stop_idx
            output_path = (
                result_dir / f"preds_{data_name}_{start_idx}-{stop_idx}.jsonl"  # noqa
            )

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Start index: {args.start_idx}")
        print(f"Stop index: {stop_idx}")
        print(f"Verbose: {args.verbose}")
        print(f"Max tokens: {max_tokens}")
        for i in range(start_idx, stop_idx):
            eg = examples[i]
            input_text = create_prompt(eg, data_name, model_name, args.data_dir)
            print(f"====== {data_name} Example {i}/{stop_idx - start_idx + 1} ======")
            pred = get_pred(
                model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose
            )
            if args.verbose:
                print(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)

    if dist.get_world_size() > 1:
        dist.barrier()
    print('Evaluation finished.')
