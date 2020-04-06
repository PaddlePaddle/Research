#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""hello world"""
import os
import sys
import json
import argparse


def read_by_lines(path, encoding="utf-8"):
    """read the data by line"""
    result = list()
    with open(path, "r") as infile:
        for line in infile:
            result.append(line.strip().decode(encoding))
    return result


def write_by_lines(path, data, t_code="utf-8"):
    """write the data"""
    with open(path, "w") as outfile:
        [outfile.write(d.encode(t_code) + "\n") for d in data]


def data_process(path, model="trigger", is_predict=False):
    """data_process"""

    def label_data(data, start, l, _type):
        """label_data"""
        for i in range(start, start + l):
            suffix = u"B-" if i == start else u"I-"
            data[i] = u"{}{}".format(suffix, _type)
        return data

    sentences = []
    output = [u"text_a"] if is_predict else [u"text_a\tlabel"]
    with open(path) as f:
        for line in f:
            d_json = json.loads(line.strip().decode("utf-8"))
            _id = d_json["id"]
            text_a = [
                u"，" if t == u" " or t == u"\n" or t == u"\t" else t
                for t in list(d_json["text"].lower())
            ]
            if is_predict:
                sentences.append({"text": d_json["text"], "id": _id})
                output.append(u'\002'.join(text_a))
            else:
                if model == u"trigger":
                    labels = [u"O"] * len(text_a)
                    for event in d_json["event_list"]:
                        event_type = event["event_type"]
                        start = event["trigger_start_index"]
                        trigger = event["trigger"]
                        labels = label_data(labels, start,
                                            len(trigger), event_type)
                    output.append(u"{}\t{}".format(u'\002'.join(text_a),
                                                   u'\002'.join(labels)))
                elif model == u"role":
                    for event in d_json["event_list"]:
                        labels = [u"O"] * len(text_a)
                        for arg in event["arguments"]:
                            role_type = arg["role"]
                            argument = arg["argument"]
                            start = arg["argument_start_index"]
                            labels = label_data(labels, start,
                                                len(argument), role_type)
                        output.append(u"{}\t{}".format(u'\002'.join(text_a),
                                                       u'\002'.join(labels)))
    if is_predict:
        return sentences, output
    else:
        return output


def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if u"B-{}".format(_type) not in labels:
            labels.extend([u"B-{}".format(_type), u"I-{}".format(_type)])
        return labels

    labels = []
    with open(path) as f:
        for line in f:
            d_json = json.loads(line.strip().decode("utf-8"))
            if model == u"trigger":
                labels = label_add(labels, d_json["event_type"])
            elif model == u"role":
                for role in d_json["role_list"]:
                    labels = label_add(labels, role["role"])
    labels.append(u"O")
    return labels


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret


def predict_data_process(trigger_file, role_file, schema_file, save_path):
    """predict_data_process"""
    pred_ret = []
    trigger_datas = read_by_lines(trigger_file)
    role_datas = read_by_lines(role_file)
    schema_datas = read_by_lines(schema_file)
    schema = {}
    for s in schema_datas:
        d_json = json.loads(s)
        schema[d_json["event_type"]] = [r["role"] for r in d_json["role_list"]]
    # 将role数据进行处理
    sent_role_mapping = {}
    for d in role_datas:
        d_json = json.loads(d)
        r_ret = extract_result(d_json["text"], d_json["labels"])
        role_ret = {}
        for r in r_ret:
            role_type = r["type"]
            if role_type not in role_ret:
                role_ret[role_type] = []
            role_ret[role_type].append(u"".join(r["text"]))
        sent_role_mapping[d_json["id"]] = role_ret

    for d in trigger_datas:
        d_json = json.loads(d)
        t_ret = extract_result(d_json["text"], d_json["labels"])
        pred_event_types = list(set([t["type"] for t in t_ret]))
        event_list = []
        for event_type in pred_event_types:
            role_list = schema[event_type]
            arguments = []
            for role_type, ags in sent_role_mapping[d_json["id"]].items():
                if role_type not in role_list:
                    continue
                for arg in ags:
                    if len(arg) == 1:
                        # 一点小trick
                        continue
                    arguments.append({"role": role_type, "argument": arg})
            event = {"event_type": event_type, "arguments": arguments}
            event_list.append(event)
        pred_ret.append({
            "id": d_json["id"],
            "text": d_json["text"],
            "event_list": event_list
        })
    pred_ret = [json.dumps(r, ensure_ascii=False) for r in pred_ret]
    write_by_lines(save_path, pred_ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Official evaluation script for DuEE version 0.1.")
    parser.add_argument(
        "--trigger_file",
        help="trigger model predict data path",
        required=True)
    parser.add_argument(
        "--role_file", help="role model predict data path", required=True)
    parser.add_argument(
        "--schema_file", help="schema file path", required=True)
    parser.add_argument("--save_path", help="save file path", required=True)
    args = parser.parse_args()
    predict_data_process(args.trigger_file, args.role_file, args.schema_file,
                         args.save_path)
