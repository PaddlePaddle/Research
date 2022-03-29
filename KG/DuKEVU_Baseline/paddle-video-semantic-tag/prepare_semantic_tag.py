# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved.
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

import os
import pandas as pd
import json
import codecs
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--trainval_path", type=str, default="dataset/dataset/train.json")
parser.add_argument(
    "--test_path", type=str, default="dataset/dataset/test_a.json")

TAG_NAMES = ["B-ENT", "I-ENT", "O"]


def gather_text_and_tags(sample, test_only=False):
    def fill_tags(surf):
        '''
        For entities that appear in text, replace their tags with 'B-ENT/I-ENT'.
        '''
        s_idx = text.find(surf)
        if s_idx != -1:
            tags[s_idx] = TAG_NAMES[0]
            for i in range(s_idx + 1, s_idx + len(surf)):
                tags[i] = TAG_NAMES[1]
            return 1
        return 0

    text = sample["title"].replace(" ", "").replace("\t", "")
    # init tag sequence with all 'O's.
    tags = [TAG_NAMES[2] for i in range(len(text))]
    entities = []
    if not test_only:
        entities = [each["@value"] for each in sample["tag"]]
    # annotate 'B-ENT' and 'I-ENT' tags.
    n_bingo_entities = sum(
        [fill_tags(surf) for surf in entities if len(surf) > 0])
    # statistics
    stats = {
        "txt_length": len(text),
        "n_entities": len(entities),
        "n_bingo_entities": n_bingo_entities,
    }
    return text, tags, stats


def stat_numberic_list(li, name="default"):
    assert isinstance(li, list)
    stat = {}
    stat["size"] = len(li)
    if all(isinstance(x, int) for x in li):
        stat["max"] = max(li)
        stat["min"] = min(li)
        stat["sum"] = sum(li)
        stat["avr"] = stat["sum"] / float(len(li))
    print("list-%s:\n\t%s" % (name, str(stat)))


def analyze_annots(stats_list):
    for key in ["txt_length", "n_entities", "n_bingo_entities"]:
        numbers = [stats[key] for stats in stats_list]
        stat_numberic_list(numbers, name=key)


def prepare_split(data, split_name, test_only=False):
    sample_lines = []
    nid_lines = []
    stats_list = []
    for idx in range(len(data)):
        text, tags, stats = gather_text_and_tags(
            data[idx], test_only=test_only)
        if len(text) == 0:
            continue
        # proper data format.
        text = '\002'.join([ch for ch in text])
        tags = '\002'.join(tags)
        sample_lines.append('\t'.join([text, tags]) + "\n")
        nid_lines.append(data[idx]["@id"] + "\n")
        stats_list.append(stats)
    if split_name == "trainval":
        # print statistics.
        analyze_annots(stats_list)

    save_split_file = "data/{}.tsv".format(split_name)
    with codecs.open(save_split_file, "w", encoding="utf-8") as ouf:
        ouf.writelines(sample_lines)
        print("Saved {}, size={}".format(save_split_file, len(sample_lines)))
    with codecs.open("data/nids.txt", "w", encoding="utf-8") as ouf:
        ouf.writelines(nid_lines)


def create_splits_indice(n_samples, SPLITS):
    assert sum([v for k, v in SPLITS]) == 1.0
    indices = list(range(n_samples))
    random.shuffle(indices)
    split2indice = {}
    r_offset = 0
    for idx, (split, ratio) in enumerate(SPLITS):
        l_offset = r_offset
        if idx == len(SPLITS) - 1:
            r_offset = n_samples
        else:
            r_offset = int(n_samples * ratio) + l_offset
        split2indice[split] = indices[l_offset:r_offset]
    return split2indice


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    random.seed(6666)

    # load data for train & validation (have labels).
    with codecs.open(args.trainval_path, "r", encoding="utf-8") as inf:
        print("Loading {}...".format(args.trainval_path))
        lines = inf.readlines()
        trainval_data = [json.loads(line) for line in lines]

    # load data for test (no labels).
    with codecs.open(args.test_path, "r", encoding="utf-8") as inf:
        print("Loading {}...".format(args.test_path))
        lines = inf.readlines()
        test_data = [json.loads(line) for line in lines]

    # split the trainval data into train-set(80%) and validation-set(20%).
    split2indice = create_splits_indice(
        len(trainval_data), [
            ("train", 3.0 / 4.0),
            ("val", 1.0 / 4.0),
        ])
    train_data = [trainval_data[idx] for idx in split2indice["train"]]
    val_data = [trainval_data[idx] for idx in split2indice["val"]]

    label_map_file = "data/label_map.json"
    with open(label_map_file, "w") as ouf:
        json.dump({tag: idx for idx, tag in enumerate(TAG_NAMES)}, ouf)
    print("Saved " + label_map_file)

    prepare_split(trainval_data, "trainval")
    prepare_split(train_data, "train")
    prepare_split(val_data, "val")
    prepare_split(test_data, "test", test_only=True)
