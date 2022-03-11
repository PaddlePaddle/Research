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
import codecs
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_path", type=str, default="dataset/dataset/test_a.json")
parser.add_argument(
    "--category_level1_result",
    type=str,
    default=osp.join("paddle-video-classify-tag",
                     "predict_results/level1_top1.json"))
parser.add_argument(
    "--category_level2_result",
    type=str,
    default=osp.join("paddle-video-classify-tag",
                     "predict_results/level2_top1.json"))
parser.add_argument(
    "--tag_result",
    type=str,
    default=osp.join("paddle-video-semantic-tag",
                     "predict_results/ents_results.json"))

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    with codecs.open(args.test_path, "r", encoding="utf-8") as inf:
        print("Loading {}...".format(args.test_path))
        lines = inf.readlines()
        nids = [json.loads(line)["@id"] for line in lines]

    # load the prediction results of 'paddle-video-classify-tag' model on test-set
    with codecs.open(
            args.category_level1_result, "r", encoding="utf-8") as inf:
        pred_level1 = json.load(inf)
    with codecs.open(
            args.category_level2_result, "r", encoding="utf-8") as inf:
        pred_level2 = json.load(inf)
    # load the prediction results of 'paddle-video-semantic-tag' model on test-set
    with codecs.open(args.tag_result, "r", encoding="utf-8") as inf:
        pred_tags = json.load(inf)

    # merge results and generate an entry for each nid.
    submission_lines = []
    for nid in nids:
        level1_category = pred_level1[nid]["class_name"] \
                          if nid in pred_level1 else ""
        level2_category = pred_level2[nid]["class_name"] \
                          if nid in pred_level2 else ""
        tags = pred_tags[nid] if nid in pred_tags else []
        result = {
            "@id": nid,
            "category": [
                {
                    "@meta": {
                        "type": "level1"
                    },
                    "@value": level1_category
                },
                {
                    "@meta": {
                        "type": "level2"
                    },
                    "@value": level2_category
                },
            ],
            "tag": [{
                "@value": tag
            } for tag in tags],
        }
        submission_lines.append(json.dumps(result, ensure_ascii=False) + "\n")

    with codecs.open("result.txt", "w", encoding="utf-8") as ouf:
        ouf.writelines(submission_lines)
    print("Saved result.txt")
