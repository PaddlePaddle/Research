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
"""
处理得到预测数据
"""
import os
import json
import sys

from utils import utils


def test_data_2_eval():
    """test_2_eval_data"""
    test_file_path = sys.argv[2]
    save_path = sys.argv[3]
    if not test_file_path or not save_path:
        raise Exception("must set test_data_path and save_path")

    datas = utils.read_by_lines(test_file_path)
    all_events = {}
    for data in datas:
        d_json = json.loads(data)
        text = d_json["text"]
        _id = utils.cal_md5(text.encode("utf-8"))
        event = {
            "trigger": d_json["trigger"],
            "trigger_start_index": d_json["trigger_start_index"],
            "event_type": d_json["event_type"],
            "class": d_json["class"],
            "arguments": d_json["arguments"],
        }
        if _id not in all_events:
            all_events[_id] = {u"id": _id, u"text": text, u"event_list": []}
        all_events[_id][u"event_list"].append(event)
    outputs = [json.dumps(x, ensure_ascii=False) for x in all_events.values()]
    utils.write_by_lines(save_path, outputs)
    print(u"test data 2 eval data, inputs {} outputs {}".format(
        len(datas), len(outputs)))


def predict_data_2_eval():
    """pred_process_with_golden_type"""
    pred_trigger_path = sys.argv[2]
    pred_role_path = sys.argv[3]
    schema_path = sys.argv[4]
    save_path = sys.argv[5]
    if not pred_trigger_path or not pred_role_path or not schema_path or not save_path:
        raise Exception(
            "must set pred_trigger_path and pred_role_path and schema_path and save_path"
        )
    print(u"predict data 2 eval data start")

    trigger_data_list = utils.read_by_lines(pred_trigger_path)
    trigger_datas = {}
    for d in trigger_data_list:
        d_json = json.loads(d)
        trigger_datas[d_json["event_id"]] = d_json
    print(u"load trigger predict datas {} from {}".format(
        len(trigger_datas), pred_trigger_path))

    role_data_list = utils.read_by_lines(pred_role_path)
    role_datas = {}
    for d in role_data_list:
        d_json = json.loads(d)
        role_datas[d_json["event_id"]] = d_json
    print(u"load role predict datas {} from {}".format(
        len(role_datas), pred_role_path))

    schema_data_list = utils.read_by_lines(schema_path)
    schema_datas = {}
    for d in schema_data_list:
        d_json = json.loads(d)
        schema_datas[d_json["event_type"]] = [
            r["role"] for r in d_json["role_list"]
        ]
    print(u"load schema datas {} from {}".format(
        len(schema_data_list), schema_path))

    all_events = {}
    for t_json in trigger_datas.values():
        text = t_json["sentence"]
        _id = utils.cal_md5(text.encode("utf-8"))
        exist_event_type = set()
        for tri_info in t_json["trigger_ret"]:
            event_type = tri_info["event_type"]
            if event_type in exist_event_type:
                continue
            trigger = tri_info["text"]
            role_type_set = set(schema_datas[event_type])

            r_json = role_datas[t_json["event_id"]]
            arguments = []
            for p_r in r_json["roles_ret"]:
                role_type = p_r["role_type"]
                if role_type in role_type_set:
                    arguments.append({
                        u"role": role_type,
                        u"argument": p_r["text"]
                    })
            if len(arguments) > 0:
                event = {
                    u"trigger": trigger,
                    u"event_type": event_type,
                    u"arguments": arguments
                }
                if _id not in all_events:
                    all_events[_id] = {
                        u"id": _id,
                        u"text": text,
                        u"event_list": []
                    }
                all_events[_id][u"event_list"].append(event)
            exist_event_type.add(event_type)
    outputs = [json.dumps(x, ensure_ascii=False) for x in all_events.values()]
    utils.write_by_lines(save_path, outputs)
    print(u"predict data 2 eval data is finished, outputs {}".format(
        len(outputs)))


def main():
    """main"""
    func_mapping = {
        "predict_data_2_eval": predict_data_2_eval,
        "test_data_2_eval": test_data_2_eval
    }
    func_name = sys.argv[1]
    if func_name not in func_name:
        raise Exception("no function {}, please choice {}".format(
            func_name, u"|".join(func_mapping.keys())))
    func_mapping[func_name]()


if __name__ == '__main__':
    main()
