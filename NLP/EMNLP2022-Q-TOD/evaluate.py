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

"""Evaluate generated response."""

import argparse
import json
import re
import string

from paddlenlp.metrics import BLEU


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["SMD", "CamRest", "MultiWOZ"], required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--entity_file", type=str, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    args = parser.parse_args()
    return args


def evaluate(args):
    """Main evaluation function."""
    with open(args.pred_file, "r") as fin:
        data = json.load(fin)
        print(f"Load prediction file from: {args.pred_file}")

    preds = []
    refs = []
    for dial in data:
        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                if args.dataset == "MultiWOZ":
                    preds.append(preprocess_text(turn["generated_response"]))
                else:
                    preds.append(turn["generated_response"])
                refs.append(turn["utterance"])
    assert len(preds) == len(refs), f"{len(preds)} != {len(refs)}"

    bleu_metric = BLEUMetric()
    entity_metric = EntityMetric(args)
    bleu_res = bleu_metric.evaluate(preds, refs)
    entity_res = entity_metric.evaluate(preds, refs)
    results = {
        "BLEU": bleu_res,
        "Entity-F1": entity_res
    }

    print(json.dumps(results, indent=2))
    with open(args.save_file, "w") as fout:
        json.dump(results, fout, indent=2)
    return


class BLEUMetric(object):
    """BLEU Metric for Response."""

    def __init__(self):
        self.metric = BLEU()

    def evaluate(self, preds, refs):
        preds, refs = self._process_text(preds, refs)
        for pred, ref in zip(preds, refs):
            self.metric.add_inst(pred, ref)
        bleu = self.metric.score()
        return bleu

    def _process_text(self, preds, refs):
        _preds = [pred.strip().lower().split(" ") for pred in preds]
        _refs = [[ref.strip().lower().split(" ")] for ref in refs]
        return _preds, _refs


class EntityMetric(object):
    """Entity Metric for Response."""

    def __init__(self, args):
        self.dataset = args.dataset
        self.entities = self._load_entities(args.entity_file)

    def evaluate(self, preds, refs):
        extracted_preds_entities = []
        extracted_refs_entities = []
        for pred, ref in zip(preds, refs):
            pred_entities = self._extract_entities(pred)
            ref_entities = self._extract_entities(ref)
            extracted_preds_entities.append(pred_entities)
            extracted_refs_entities.append(ref_entities)
        entity_f1 = self._compute_entity_f1(extracted_preds_entities, extracted_refs_entities)
        return entity_f1

    def _load_entities(self, entities_file):
        with open(entities_file, "r") as fin:
            raw_entities = json.load(fin)
        entities = set()

        if self.dataset == "SMD":
            for slot, values in raw_entities.items():
                for val in values:
                    if slot == "poi":
                        entities.add(val["address"])
                        entities.add(val["poi"])
                        entities.add(val["type"])
                    elif slot == "distance":
                        entities.add(f"{val} miles")
                    elif slot == "temperature":
                        entities.add(f"{val}f")
                    else:
                        entities.add(val)

            # add missing entities
            missed_entities = ["yoga", "tennis", "swimming", "football", " lab ", "doctor", "optometrist", "dentist",
                               "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                               "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th", "jill", "jack", " hr "]
            for missed_entity in missed_entities:
                entities.add(missed_entity)
            # special handle of "hr"
            entities.remove("hr")

        else:
            for slot, values in raw_entities.items():
                for val in values:
                    if self.dataset == "MultiWOZ" and slot == "choice":
                        val = f"choice-{val}"
                    entities.add(val)

        processed_entities = []
        for val in entities:
            processed_entities.append(val.lower())
        processed_entities.sort(key=lambda x: len(x), reverse=True)
        return processed_entities

    def _extract_entities(self, response):
        def _is_sub_str(str_list, sub_str):
            for str_item in str_list:
                if sub_str in str_item:
                    return True
            return False

        response = f" {response} ".lower()
        extracted_entities = []

        if self.dataset == "SMD":
            # preprocess response
            for h in range(0, 13):
                response = response.replace(f"{h} am", f"{h}am")
                response = response.replace(f"{h} pm", f"{h}pm")
            for low_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for high_temp in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    response = response.replace(f"{low_temp}-{high_temp}f", f"{low_temp}f-{high_temp}f")

        for entity in self.entities:
            if self.dataset == "MultiWOZ":
                success_tag = False
                if entity.startswith("choice-"):
                    entity = entity[7:]
                    if entity == "many":
                        if entity in re.sub(r"(many (other types|food types|cuisines)|how many)", " ", response):
                            success_tag = True
                    elif entity == "all":
                        if re.search(r"all (of the|expensive|moderate|cheap)", response):
                            success_tag = True
                    elif entity == "to":
                        success_tag = False
                    else:
                        if re.search(f"(there are|there is|found|have about|have)( only|) {entity}", response):
                            success_tag = True
                elif entity == "centre":
                    if entity in response.replace("cambridge towninfo centre", " "):
                        success_tag = True
                elif entity == "free":
                    if re.search(r"free (parking|internet|wifi)", response):
                        success_tag = True
                elif entity in response or entity.lower() in response.lower():
                    success_tag = True

                if success_tag:
                    extracted_entities.append(entity)
                    response = response.replace(entity, " ")

            else:
                if entity in response and not _is_sub_str(extracted_entities, entity):
                    extracted_entities.append(entity)

        return extracted_entities

    def _compute_entity_f1(self, preds, refs):
        """Compute Entity-F1."""
        def _count(pred, ref):
            tp, fp, fn = 0, 0, 0
            if len(ref) != 0:
                for g in ref:
                    if g in pred:
                        tp += 1
                    else:
                        fn += 1
                for p in set(pred):
                    if p not in ref:
                        fp += 1
            return tp, fp, fn

        tp_all, fp_all, fn_all = 0, 0, 0
        for pred, ref in zip(preds, refs):
            tp, fp, fn = _count(pred, ref)
            tp_all += tp
            fp_all += fp
            fn_all += fn

        precision = tp_all / float(tp_all + fp_all) if (tp_all + fp_all) != 0 else 0
        recall = tp_all / float(tp_all + fn_all) if (tp_all + fn_all) != 0 else 0
        f1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return f1


def preprocess_text(text):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text


if __name__ == "__main__":
    args = setup_args()
    evaluate(args)
