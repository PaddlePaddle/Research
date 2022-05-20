#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
File: classifier_grounded.py
Author: liwei(liwei85@baidu.com)
Date: 2021-09-15 14:22
Desc: Model for classifier.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import paddle
import paddle.fluid as fluid
from finetune import glue_eval
from collections import OrderedDict
from utils.utils import print_eval_log

from model.unimo_grounded import VlModel as GroundingModel
from model.unimo_grounded_baseline import VlModel as BaselineModel


def create_model(args, pyreader_name, vl_config, is_prediction=False):
    stype = 'int64'
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, 1, args.max_seq_len], [-1, 1],
                [-1, 1]],
        dtypes=[stype, stype, stype, 'float32', stype, stype],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, text_mask, labels, qids) = fluid.layers.read_file(pyreader)
    text_input = {"text.word_embedding": src_ids, "text.sent_embedding": sent_ids, "text.pos_embedding": pos_ids}

    text_enc_layers = [int(i) for i in args.text_enc_layers.split(',')]
    grounding_enc_layers = [int(i) for i in args.grounding_enc_layers.split(',')]

    if args.model_type == "grounded":
        model = GroundingModel(text_input=text_input,
                               text_mask=text_mask,
                               config=vl_config,
                               weight_sharing=args.weight_sharing,
                               grounding_method=args.grounding_method,
                               topk_value=args.topk_value,
                               with_grounding_projection=args.with_grounding_projection,
                               with_grounding_pos=args.with_grounding_pos,
                               text_enc_layers=text_enc_layers,
                               grounding_enc_layers=grounding_enc_layers)
    elif args.model_type == "baseline":
        model = BaselineModel(text_input=text_input,
                              text_mask=text_mask,
                              config=vl_config,
                              weight_sharing=args.weight_sharing,
                              text_enc_layers=text_enc_layers,
                              grounding_enc_layers=grounding_enc_layers)
    else:
        raise ValueError("The model_type is invalid!!!")

    cls_feats = model.get_pooled_txt_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")

    if args.base_mnli:
        cls_params_name = ["cls_out_new_w", "cls_out_new_b"]
    else:
        cls_params_name = ["cls_out_w", "cls_out_b"]

    logits = fluid.layers.fc(
        input=cls_feats,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            name=cls_params_name[0],
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name=cls_params_name[1], initializer=fluid.initializer.Constant(0.)))

    if is_prediction:
        probs = fluid.layers.softmax(logits)
        feed_targets_name = [
            src_ids.name, pos_ids.name, sent_ids.name, text_mask.name
        ]
        return pyreader, probs, feed_targets_name

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)

    loss = fluid.layers.mean(x=ce_loss)

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=labels, total=num_seqs)

    graph_vars = {
        "loss": loss,
        "probs": probs,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs,
        "qids": qids
    }

    # for k, v in graph_vars.items():
    # v.persistable = True

    return pyreader, graph_vars


def predict(exe, test_program, test_pyreader, graph_vars, dev_count=1):
    test_pyreader.start()
    qids, scores, probs, preds = [], [], [], []
    fetch_list = [graph_vars["probs"].name, graph_vars["qids"].name]
    while True:
        try:
            if dev_count == 1:
                np_probs, np_qids = exe.run(program=test_program, fetch_list=fetch_list)
            else:
                np_probs, np_qids = exe.run(fetch_list=fetch_list)

            qids.extend(np_qids.reshape(-1).tolist())
            np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            preds.extend(np_preds)
            probs.append(np_probs)

        except fluid.core.EOFException:
            test_pyreader.reset()
            break

    probs = np.concatenate(probs, axis=0).reshape([len(qids), -1])
    return qids, preds, probs


def evaluate(args, exe, test_program, test_pyreader, graph_vars, eval_phase):
    total_cost, total_num_seqs = 0.0, 0.0
    qids, labels, scores, preds = [], [], [], []

    test_pyreader.start()
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name,
        graph_vars["probs"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name
    ]
    while True:
        try:
            np_loss, np_probs, np_labels, np_num_seqs, np_qids = exe.run(
                program=test_program, fetch_list=fetch_list) \
                if not args.use_multi_gpu_test else exe.run(fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is not None:
                qids.extend(np_qids.reshape(-1).tolist())
            scores.extend(np_probs[:, 1].reshape(-1).tolist())
            np_preds = list(np.argmax(np_probs, axis=1).astype(np.float32))
            preds.extend([float(val) for val in np_preds])
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    ret = OrderedDict()
    ret['phase'] = eval_phase
    ret['loss'] = round(total_cost / total_num_seqs, 4)
    ret['data_num'] = total_num_seqs
    ret['used_time'] = round(time_end - time_begin, 4)

    metrics = OrderedDict()
    metrics["acc_and_f1"] = glue_eval.acc_and_f1
    metrics["simple_accuracy"] = glue_eval.simple_accuracy
    metrics["matthews_corrcoef"] = glue_eval.matthews_corrcoef

    if args.eval_mertrics in metrics:
        ret_metric = metrics[args.eval_mertrics](preds, labels)
        ret.update(ret_metric)
        print_eval_log(ret)
    else:
        raise ValueError('unsupported metric {}'.format(args.eval_mertrics))

    return ret
