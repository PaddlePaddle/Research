from argparse import ArgumentParser, Namespace
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help="Which task to use. Note that \"all\" can only be used in `compute_scores.py`.",  # noqa
    )
    p.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help="The directory of data."
    )
    p.add_argument("--output_dir", type=str, default="../results", help="Where to dump the prediction results.")  # noqa
    p.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="The path of the model (in HuggingFace (HF) style). If specified, it will try to load the model from the specified path, else, it wll default to the official HF path.",  # noqa
    )  # noqa
    p.add_argument(
        "--model_name",
        type=str,
        choices=["llama3-8b"],
        default="llama3-8b",
        help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    )
    p.add_argument("--start_idx", type=int, default=0, help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.")  # noqa
    p.add_argument("--stop_idx", type=int, help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.")  # noqa
    p.add_argument("--verbose", action='store_true')
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--enable_nacl_evict", action='store_true', help="Enable the NACL eviction strategy.") # noqa
    p.add_argument("--proxy_tokens_ratio", type=float, default=0.01, help="Proportion of proxy tokens in the sequence length, used for calculating the Attention Score.") # noqa
    p.add_argument("--proxy_token_keep_ratio", type=float, default=0.12, help="Ratio of tokens retained by the proxy token eviction strategy relative to sequence length.") # noqa
    p.add_argument("--random_token_keep_ratio", type=float, default=0.07, help="Ratio of tokens retained by the random eviction strategy relative to sequence length.") # noqa
    p.add_argument("--token_protect_ratio", type=float, default=0.01, help="Ratio of tokens retained for protection, including sink tokens and recently used tokens, relative to sequence length.") # noqa
    p.add_argument("--sink_tokens", type=int, default=256, help="Number of sink tokens.") # noqa
    p.add_argument("--min_eviction_seqlen", type=int, default=2048, help="Minimum sequence length to trigger the eviction strategy.") # noqa
    return p.parse_args()
