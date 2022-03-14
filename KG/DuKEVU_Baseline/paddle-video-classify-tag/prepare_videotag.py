# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved.
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
import os.path as osp
import requests
import time
import codecs
import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument(
    "--trainval_path", type=str, default="dataset/dataset/train.json")
parser.add_argument(
    "--test_path", type=str, default="dataset/dataset/test_a.json")
parser.add_argument(
    "--trainval_tsn_feature_dir",
    type=str,
    default="dataset/tsn_features_train")
parser.add_argument(
    "--test_tsn_feature_dir",
    type=str,
    default="dataset/tsn_features_test_a")


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


def prepare_split(data, split_name, test_only=False, gather_labels=False):
    '''
      1. Prepare ALL (unique) labels for classification from trainval-set.
      2. For each split, generate sample list for level1 & level2 classification.
    '''
    sample_nids = [sample["@id"] for sample in data]
    level1_labels = []
    level2_labels = []
    if not test_only:
        for sample in data:
            category = {
                each["@meta"]["type"]: each["@value"]
                for each in sample["category"]
            }
            level1_labels.append(category["level1"])
            level2_labels.append(category["level2"])

    def create_sample_list(sample_labels, level_name):
        save_label_file = "data/{}_label.txt".format(level_name)
        if gather_labels:
            # For trainval set:
            # Gather candidate labels and dump to {level1,level2}_label.txt
            labels = sorted([str(label) for label in list(set(sample_labels))])
            with codecs.open(save_label_file, "w", encoding="utf-8") as ouf:
                ouf.writelines([label + "\n" for label in labels])
                print("Saved " + save_label_file)
        else:
            # For test set: load existing labels.
            with codecs.open(save_label_file, "r", encoding="utf-8") as inf:
                labels = [line.strip() for line in inf.readlines()]
        label2idx = {label: idx for idx, label in enumerate(labels)}
        sample_lines = []
        # Generate sample list: one sample per line (feature_path -> label)
        for i in range(len(sample_nids)):
            label_indice = label2idx[str(sample_labels[i])] if not test_only \
                           else -1
            if split_name in ["train", "val", "trainval"]:
                tsn_feature_dir = args.trainval_tsn_feature_dir
            elif split_name in ["test"]:
                tsn_feature_dir = args.test_tsn_feature_dir
            feature_path = osp.join(tsn_feature_dir,
                                    "{}.npy".format(sample_nids[i]))
            if osp.exists(feature_path):
                line = "{} {}\n".format(feature_path, str(label_indice))
                sample_lines.append(line)
        save_split_file = "data/{}_{}.list".format(level_name, split_name)
        with codecs.open(save_split_file, "w", encoding="utf-8") as ouf:
            ouf.writelines(sample_lines)
            print("Saved {}, size={}".format(save_split_file,
                                             len(sample_lines)))

    create_sample_list(level1_labels, "level1")
    create_sample_list(level2_labels, "level2")


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
            ("train", 4.0 / 5.0),
            ("val", 1.0 / 5.0),
        ])
    train_data = [trainval_data[idx] for idx in split2indice["train"]]
    val_data = [trainval_data[idx] for idx in split2indice["val"]]

    prepare_split(trainval_data, "trainval", gather_labels=True)
    prepare_split(train_data, "train")
    prepare_split(val_data, "val")
    prepare_split(test_data, "test", test_only=True)
"""
# The download links of these videos are broken and have NO tsn features.
failure_nids = set([
    "5503534586756728779", "5484123847203473111", "5410347660659009105", "5370913341167173620", "5361502844374323563", "5350078935908955482", "5340634113699834265", "5271069752487955873", "5230444730564750890", "5194509976461059160", "5118222917337926016", "5103160273644731677", "5074443564062974347", "5065400231195314112", "5006552169895847911", "4919681259539538890", "4905984166294562821", "4864450389741796030", "4859124879129192692", "4855974464706903421", "4845559551333371675", "4828888150406334357", "4824792564178939105", "4802768573242885647", "4792223380614790014", "4738992990488461686", "4724936442653803140", "4715585255357600897", "4692163870807479317", "4658942101189354026", "4658653909052614587", "4657820234470843275", "4621317307544048312", "4618030660885379923", "4574309856843075040", "4533137436743815117", "4500977868466523673", "4466964313051292443", "4459163102065788675", "4441834111522674708", "4430231493408238493", "4421806911125438202", "4379622845966654686", "4379258559956993017", "4378312992758640298", "4355267417914093000", "4313257118884501482", "4310768623161889718", "4283985456249430334", "4266460074173303169", "4254437176427137581", "4247865601814949854", "4234962686877614203", "4226207597954384926", "4100296203440702477", "4093766203795855445", "4065903213316100763", "3922591484788577150", "3841506779949374890", "3817440398943197233", "3717763746818689975", "3716673276330887681", "3644764215119656352", "3636427022179216663", "3539399068230872342", "3474796643844118991", "3421091986050159712", "2947537460713120464", "6861301171717054449", "119183731142996653"])
"""
