# coding: utf-8
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
"""sequence labeling model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import json
import argparse
import numpy as np
import multiprocessing

import paddle
import logging
import paddle.fluid as fluid

from six.moves import xrange

from model.ernie import ErnieModel

log = logging.getLogger(__name__)


def create_model(args, pyreader_name, ernie_config, is_prediction=False):
    """func"""
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1], [-1]],
        dtypes=[
            'int64', 'int64', 'int64', 'int64', 'float32', 'int64', 'int64'
        ],
        lod_levels=[0, 0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, sent_ids, pos_ids, task_ids, input_mask, labels,
     seq_lens) = fluid.layers.read_file(pyreader)

    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=pos_ids,
        sentence_ids=sent_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=ernie_config,
        use_fp16=args.use_fp16)

    enc_out = ernie.get_sequence_output()

    emission = fluid.layers.fc(
        input=enc_out,
        size=args.num_labels,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Uniform(
                low=-0.1, high=0.1),
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=1e-4)),
        num_flatten_dims=2)

    crf_cost = fluid.layers.linear_chain_crf(
        input=emission,
        label=labels,
        param_attr=fluid.ParamAttr(
            name='crfw', learning_rate=args.crf_learning_rate),
        length=seq_lens)

    loss = fluid.layers.mean(x=crf_cost)

    crf_decode = fluid.layers.crf_decoding(
        input=emission,
        param_attr=fluid.ParamAttr(name='crfw'),
        length=seq_lens)

    lod_labels = fluid.layers.squeeze(labels, axes=[-1])

    num_chunk_types = (
        (args.num_labels - 1) // (len(args.chunk_scheme) - 1))  # IOB配置

    (_, _, _, num_infer, num_label, num_correct) = fluid.layers.chunk_eval(
        input=crf_decode,
        label=lod_labels,
        chunk_scheme=args.chunk_scheme,
        num_chunk_types=num_chunk_types,
        seq_length=seq_lens)
    """
    enc_out = fluid.layers.dropout(x=enc_out,
                                   dropout_prob=0.1,
                                   dropout_implementation="upscale_in_train")

    logits = fluid.layers.fc(
        input=enc_out,
        size=args.num_labels,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_seq_label_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(name="cls_seq_label_out_b",
                                  initializer=fluid.initializer.Constant(0.)))

    infers = fluid.layers.argmax(logits, axis=2)
    ret_infers = fluid.layers.reshape(x=infers, shape=[-1, 1])
    lod_labels = fluid.layers.sequence_unpad(labels, seq_lens)
    lod_infers = fluid.layers.sequence_unpad(infers, seq_lens)

    num_chunk_types = (
        (args.num_labels - 1) // (len(args.chunk_scheme) - 1))  # IOB配置

    (_, _, _, num_infer, num_label,
     num_correct) = fluid.layers.chunk_eval(input=lod_infers,
                                            label=lod_labels,
                                            chunk_scheme=args.chunk_scheme,
                                            num_chunk_types=num_chunk_types)

    labels = fluid.layers.flatten(labels, axis=2)
    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=fluid.layers.flatten(logits, axis=2),
        label=labels,
        return_softmax=True)
    input_mask = fluid.layers.flatten(input_mask, axis=2)
    ce_loss = ce_loss * input_mask
    loss = fluid.layers.mean(x=ce_loss)
    """

    graph_vars = {
        "inputs": src_ids,
        "loss": loss,
        "seqlen": seq_lens,
        "crf_decode": crf_decode,
        "num_infer": num_infer,
        "num_label": num_label,
        "num_correct": num_correct,
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def calculate_f1(num_label, num_infer, num_correct):
    """calculate_f1"""
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(exe, program, pyreader, graph_vars, tag_num, dev_count=1):
    """func"""
    fetch_list = [
        graph_vars["num_infer"].name, graph_vars["num_label"].name,
        graph_vars["num_correct"].name
    ]

    total_label, total_infer, total_correct = 0.0, 0.0, 0.0
    time_begin = time.time()
    pyreader.start()
    while True:
        try:
            np_num_infer, np_num_label, np_num_correct = exe.run(
                program=program, fetch_list=fetch_list)
            total_infer += np.sum(np_num_infer)
            total_label += np.sum(np_num_label)
            total_correct += np.sum(np_num_correct)

        except fluid.core.EOFException:
            pyreader.reset()
            break

    precision, recall, f1 = calculate_f1(total_label, total_infer,
                                         total_correct)
    return precision, recall, f1


def parse_crf_ret(np_inputs, crf_decodes, np_lens):
    """parse_crf_ret"""
    np_inputs = np.squeeze(np_inputs)
    out = []
    for index in range(len(np_lens)):
        src_ids = [_id for _id in np_inputs[index][1:np_lens[index] - 1]]
        tag_ids = [_id for _id in crf_decodes[index][1:np_lens[index] - 1]]
        out.append((list(src_ids), list(tag_ids)))
    return out


def predict(exe, test_program, test_pyreader, graph_vars, dev_count=1):
    """func"""
    fetch_list = [
        graph_vars["inputs"].name,
        graph_vars["crf_decode"].name,
        graph_vars["seqlen"].name,
    ]

    test_pyreader.start()
    res = []
    while True:
        try:
            inputs, crf_decodes, np_lens = exe.run(program=test_program,
                                                   fetch_list=fetch_list)
            #r = chunk_predict(inputs, probs, np_lens, dev_count)
            res += parse_crf_ret(inputs, crf_decodes, np_lens)
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    return res
