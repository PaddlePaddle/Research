#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main inference program."""

import argparse
import json
import os
import string

import paddle
from paddlenlp.transformers import generation_utils, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm, trange


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--save_file", type=str, required=True)
    args = parser.parse_args()
    return args


def infer(args):
    """Main inference function."""
    with open(args.infer_file, "r") as fin:
        data = json.load(fin)
        print(f"Load inference file from: {args.infer_file}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.eval()
    model.to("gpu:0")
    print(f"Load model from: {args.model_path}")

    query_generation(data, tokenizer, model, args.batch_size, args.beam_size)
    knowledge_retrieval(data, args.batch_size)
    response_generation(data, tokenizer, model, args.batch_size, args.beam_size)

    if not os.path.exists(os.path.dirname(args.save_file)):
        os.makedirs(os.path.dirname(args.save_file))
    with open(args.save_file, "w") as fout:
        json.dump(data, fout, indent=2)
        print(f"Save inference output to: {args.save_file}")
    return


def query_generation(data, tokenizer, model, batch_size, beam_size):
    """Generate query by dialogue context."""
    # prepare input samples
    samples = []
    for dial in data:
        context = []
        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                src = "translate dialogue context to query : " + " | ".join(context)
                samples.append(src)
            utt = preprocess_text(turn["utterance"])
            context.append(utt)

    # call Q-TOD model
    generated_queries = generate(samples, tokenizer, model, batch_size, beam_size)

    # save generated query into data
    sample_idx = 0
    for dial in data:
        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                turn["generated_query"] = generated_queries[sample_idx]
                sample_idx += 1

    return


def knowledge_retrieval(data, batch_size):
    """Retrieve relevant knowledge by generated query."""
    import rocketqa
    model = rocketqa.load_model(model="v2_marco_ce", use_cuda=True, device_id=0, batch_size=batch_size)

    for dial in tqdm(data, desc="Retrieval"):
        excluded_fields = ["id", "location", "type"] if dial["scenario"]["kb"]["kb_title"] == "camrest676" else []
        fields = [name for name in dial["scenario"]["kb"]["column_names"] if name not in excluded_fields]
        knowledge_records = dial["scenario"]["kb"]["items"] or []
        knowledge_seqs = [linearize_knowledge_record(k, fields) for k in knowledge_records]

        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                if len(knowledge_records) == 0:
                    turn["retrieved_knowledge"] = []
                else:
                    queries = [turn["generated_query"]] * len(knowledge_records)
                    scores = model.matching(query=queries, para=knowledge_seqs)
                    _, sorted_records = zip(*sorted(zip(scores, knowledge_records), key=lambda x: x[0], reverse=True))
                    turn["retrieved_knowledge"] = list(sorted_records)[:3]

    # RocketQA uses PaddlePaddle static mode but PaddleNLP uses dynamic mode
    paddle.disable_static()
    return


def response_generation(data, tokenizer, model, batch_size, beam_size):
    """Generate system response by dialogue context and retrieved knowledge."""
    # prepare input samples
    samples = []
    for dial in data:
        excluded_fields = ["id", "location", "type"] if dial["scenario"]["kb"]["kb_title"] == "camrest676" else []
        fields = [name for name in dial["scenario"]["kb"]["column_names"] if name not in excluded_fields]
        context = []

        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                retrieved_knowledge_seq = linearize_knowledge(turn["retrieved_knowledge"], fields)
                src = "generate system response based on knowledge and dialogue context : knowledge : " + \
                    retrieved_knowledge_seq + " ; dialogue context : " + " | ".join(context)
                samples.append(src)
            utt = preprocess_text(turn["utterance"])
            context.append(utt)

    # call Q-TOD model
    generated_responses = generate(samples, tokenizer, model, batch_size, beam_size)

    # save generated response into data
    sample_idx = 0
    for dial in data:
        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                turn["generated_response"] = generated_responses[sample_idx]
                sample_idx += 1

    return


def preprocess_text(text):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text


def linearize_knowledge_record(knowledge_record, fields):
    """Convert a knowledge record into a flatten sequence with special symbols."""
    knowledge_seq = []
    for f in fields:
        value = preprocess_text(str(knowledge_record.get(f, "")))
        knowledge_seq.append(f.replace("_", " ") + " : " + value)
    return " | ".join(knowledge_seq)


def linearize_knowledge(knowledge, fields):
    """Convert knowledge into a flatten sequecen with special symbols."""
    knowledge_seq = []
    knowledge_seq.append("col : " + " | ".join(map(lambda x: x.replace("_", " "), fields)))
    for idx, record in enumerate(knowledge):
        values = []
        for f in fields:
            v = preprocess_text(str(record.get(f, "")))
            values.append(v)

        record_seq = " | ".join(values)
        knowledge_seq.append(f"row {idx} : {record_seq}")
    return " || ".join(knowledge_seq)


@paddle.no_grad()
def generate(samples, tokenizer, model, batch_size, beam_size):
    """Call Q-TOD model generation."""
    outputs = []
    for idx in trange(0, len(samples), batch_size, desc="Generation"):
        batch = samples[idx: idx + batch_size]
        tokenized_batch = tokenizer(
            batch,
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pd"
        )
        batch_out, _ = model.generate(
            tokenized_batch["input_ids"],
            decode_strategy="beam_search",
            num_beams=beam_size,
            max_length=128,
            length_penalty=1,
            attention_mask=tokenized_batch.get("attention_mask")
        )
        batch_pred = tokenizer.batch_decode(batch_out, skip_special_tokens=True)

        outputs.extend(batch_pred)
    return outputs


class FixedBeamHypotheses:
    """Fix length penalty difference."""

    def __init__(self, num_beams, length_penalty, early_stopping):
        """Initialize n-best list of hypotheses."""
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """Number of hypotheses in the list."""
        return len(self.beams)

    def add(self, hyp, sum_logprobs, origin_len=0):
        """Add a new hypothesis to the list."""
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len, origin_len=0):
        """
        If there are enough hypotheses and that none of the hypotheses being 
        generated can become better than the worst one in the heap, then we 
        are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / (cur_len ** self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret


if __name__ == "__main__":
    generation_utils.BeamHypotheses = FixedBeamHypotheses
    args = setup_args()
    infer(args)
