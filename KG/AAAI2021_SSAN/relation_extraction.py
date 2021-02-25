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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


import os
import time
import argparse
import numpy as np
import multiprocessing

import paddle
import logging
import paddle.fluid as fluid
from paddle.fluid.layers import core

from six.moves import xrange

from model.SSAN import ErnieModel

log = logging.getLogger(__name__)


def create_model(args, pyreader_name, ernie_config, is_prediction=False):
    src_ids = fluid.layers.data(name='1', shape=[args.batch_size, args.max_seq_len, 1], append_batch_size=False, dtype='int64')
    input_mask = fluid.layers.data(name='2', shape=[args.batch_size, args.max_seq_len, 1], append_batch_size=False, dtype='float32')
    sent_ids = fluid.layers.data(name='3', shape=[args.batch_size, args.max_seq_len, 1], append_batch_size=False, dtype='int64')
    pos_ids = fluid.layers.data(name='4', shape=[args.batch_size, args.max_seq_len, 1], append_batch_size=False, dtype='int64')
    task_ids = fluid.layers.data(name='5', shape=[args.batch_size, args.max_seq_len, 1], append_batch_size=False, dtype='int64')
    ent_masks = fluid.layers.data(name='6', shape=[args.batch_size, args.max_ent_cnt, args.max_seq_len], append_batch_size=False, dtype='float32')
    label_ids = fluid.layers.data(name='7', shape=[args.batch_size, args.max_ent_cnt, args.max_ent_cnt, args.num_labels], append_batch_size=False, dtype='int64')
    label_masks = fluid.layers.data(name='8', shape=[args.batch_size, args.max_ent_cnt, args.max_ent_cnt], append_batch_size=False, dtype='float32')
    ent_ner = fluid.layers.data(name='9', shape=[args.batch_size, args.max_seq_len, 1], append_batch_size=False, dtype='int64')
    ent_pos = fluid.layers.data(name='10', shape=[args.batch_size, args.max_seq_len, 1], append_batch_size=False, dtype='int64')
    ent_distance = fluid.layers.data(name='11', shape=[args.batch_size, args.max_ent_cnt, args.max_ent_cnt, 1], append_batch_size=False, dtype='int64')
    structure_mask = fluid.layers.data(name='12', shape=[args.batch_size, 5, args.max_seq_len, args.max_seq_len], append_batch_size=False, dtype='float32')

    pyreader = fluid.io.DataLoader.from_generator(feed_list=[src_ids, input_mask, sent_ids, pos_ids, task_ids, ent_masks, label_ids,
                                                             label_masks, ent_ner, ent_pos, ent_distance, structure_mask],
                                                  capacity=70,
                                                  iterable=False)

    src_ids = fluid.layers.cast(src_ids, 'int64')
    sent_ids = fluid.layers.cast(sent_ids, 'int64')
    pos_ids = fluid.layers.cast(pos_ids, 'int64')
    ent_ner = fluid.layers.cast(ent_ner, 'int64')
    ent_pos = fluid.layers.cast(ent_pos, 'int64')
    ent_distance = fluid.layers.cast(ent_distance, 'int64')

    structure_mask = fluid.layers.unsqueeze(structure_mask, [2])
    structure_mask = fluid.layers.expand(structure_mask, [1, 1, ernie_config['num_attention_heads'], 1, 1])

    ernie = ErnieModel(
        src_ids=src_ids,
        input_mask=input_mask,
        sentence_ids=sent_ids,
        position_ids=pos_ids,
        task_ids=task_ids,
        ent_ner=ent_ner,
        ent_pos=ent_pos,
        structure_mask=structure_mask,
        with_ent_structure=args.with_ent_structure,
        config=ernie_config,
        use_fp16=args.use_fp16)

    enc_out = ernie.get_sequence_output()
    enc_out = fluid.layers.fc(input=enc_out, size=128, num_flatten_dims=2,
                              param_attr=fluid.ParamAttr(
                                  name="dim_reduction_w",
                                  initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                              bias_attr=fluid.ParamAttr(
                                  name="dim_reduction_b",
                                  initializer=fluid.initializer.Constant(0.)))
    ent_rep = fluid.layers.matmul(x=ent_masks, y=enc_out)
    ent_rep_h = fluid.layers.unsqueeze(ent_rep, [2])
    ent_rep_h = fluid.layers.expand(ent_rep_h, [1, 1, args.max_ent_cnt, 1])
    ent_rep_t = fluid.layers.unsqueeze(ent_rep, [1])
    ent_rep_t = fluid.layers.expand(ent_rep_t, [1, args.max_ent_cnt, 1, 1])

    ent_distance_feature_h = fluid.layers.embedding(
        ent_distance,
        size=[20, 20],
        padding_idx=10,
        dtype=core.VarDesc.VarType.FP32,
        param_attr=fluid.ParamAttr(
            name='ent_distance_feature', initializer=fluid.initializer.TruncatedNormal(scale=0.02)))
    ent_distance_feature_t = fluid.layers.embedding(
        20 - ent_distance,
        size=[20, 20],
        padding_idx=10,
        dtype=core.VarDesc.VarType.FP32,
        param_attr=fluid.ParamAttr(
            name='ent_distance_feature', initializer=fluid.initializer.TruncatedNormal(scale=0.02)))

    ent_rep_h = fluid.layers.concat(input=[ent_rep_h, ent_distance_feature_h], axis=-1)
    ent_rep_t = fluid.layers.concat(input=[ent_rep_t, ent_distance_feature_t], axis=-1)
    ent_rep_h = fluid.layers.dropout(
        x=ent_rep_h, dropout_prob=0.1, dropout_implementation="upscale_in_train")
    ent_rep_t = fluid.layers.dropout(
        x=ent_rep_t, dropout_prob=0.1, dropout_implementation="upscale_in_train")

    ent_rep_h = fluid.layers.reshape(x=ent_rep_h, shape=[-1, ent_rep_h.shape[-1]], inplace=True)
    ent_rep_t = fluid.layers.reshape(x=ent_rep_t, shape=[-1, ent_rep_t.shape[-1]], inplace=True)

    logits = fluid.layers.bilinear_tensor_product(x=ent_rep_h, y=ent_rep_t, size=args.num_labels)
    logits = fluid.layers.reshape(x=logits, shape=[-1, args.max_ent_cnt, args.max_ent_cnt, args.num_labels], inplace=True)

    bceloss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label_ids)
    bceloss = fluid.layers.reduce_mean(bceloss, dim=-1)
    bceloss = fluid.layers.reshape(x=bceloss, shape=[-1, label_masks.shape[1], label_masks.shape[2]])
    bceloss = fluid.layers.reduce_sum(label_masks * bceloss, dim=[1, 2]) / fluid.layers.reduce_sum(label_masks, dim=[1, 2])

    loss = fluid.layers.mean(bceloss)

    logits = fluid.layers.sigmoid(logits)

    graph_vars = {
        "inputs": src_ids,
        "ent_masks": ent_masks,
        "label_ids": label_ids,
        "loss": loss,
        "logits": logits,
    }

    for k, v in graph_vars.items():
        v.persistable = True

    return pyreader, graph_vars


def batch_eval(logits, ent_masks, label_ids):

    tp, tn, fp, fn = 0, 0, 0, 0

    for ins in range(len(logits)):
        logits_ = logits[ins]
        ent_masks_ = ent_masks[ins]
        label_ids_ = label_ids[ins]
        for h in range(len(ent_masks_)):
            for t in range(len(ent_masks_)):
                if h == t:
                    continue
                if np.all(ent_masks_[h] == 0) or np.all(ent_masks_[t] == 0):
                    continue
                for r in range(len(label_ids_[0][0])):
                    if r == 0:
                        continue
                    else:
                        if logits_[h][t][r] >= 0.5 and label_ids_[h][t][r] == 1:
                            tp += 1
                        elif logits_[h][t][r] >= 0.5 and label_ids_[h][t][r] != 1:
                            fp += 1
                        elif logits_[h][t][r] <= 0.5 and label_ids_[h][t][r] == 1:
                            fn += 1
    p = tp / (tp + fp + 1e-20)
    r = tp / (tp + fn + 1e-20)
    f1 = 2 * p * r / (p + r + 1e-20)

    return f1


def evaluate(exe,
             program,
             pyreader,
             graph_vars,
             dev_examples,
             predicate_map):
    fetch_list = [
        graph_vars["loss"].name, graph_vars["logits"].name,
        graph_vars["ent_masks"].name, graph_vars["label_ids"].name
    ]
    time_begin = time.time()
    pyreader.start()
    logits_all = None
    ent_masks_all = None
    label_ids_all = None
    loss_all = 0
    while True:
        try:
            loss, logits, ent_masks, label_ids = exe.run(program=program, fetch_list=fetch_list)
            loss_all += loss
            if logits_all is None:
                logits_all = logits
                ent_masks_all = ent_masks
                label_ids_all = label_ids
            else:
                logits_all = np.append(logits_all, logits, axis=0)
                ent_masks_all = np.append(ent_masks_all, ent_masks, axis=0)
                label_ids_all = np.append(label_ids_all, label_ids, axis=0)
        except fluid.core.EOFException:
            pyreader.reset()
            break
    loss_avg = loss_all / len(dev_examples)
    total_labels = 0
    output_preds = []
    for (i, (example, logit, ent_mask)) in enumerate(zip(dev_examples, logits_all, ent_masks_all)):
        spo_gt_tmp = []
        for spo_gt in example.labels:
            spo_gt_tmp.append((spo_gt['h'], spo_gt['t'], spo_gt['r']))
        total_labels += len(spo_gt_tmp)
        for h in range(len(example.vertexSet)):
            for t in range(len(example.vertexSet)):
                if h == t:
                    continue
                if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                    continue
                for predicate_id, logit_tmp in enumerate(logit[h][t]):
                    if predicate_id == 0:
                        continue
                    if (h, t, predicate_map[predicate_id]) in spo_gt_tmp:
                        flag = True
                    else:
                        flag = False
                    output_preds.append((flag, logit_tmp, example.title, h, t, predicate_map[predicate_id]))
    output_preds.sort(key=lambda x: x[1], reverse=True)
    pr_x = []
    pr_y = []
    correct = 0
    for i, pred in enumerate(output_preds):
        correct += pred[0]
        pr_y.append(float(correct) / (i + 1))
        pr_x.append(float(correct) / total_labels)
    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = output_preds[f1_pos][1]
    output_preds_thresh = []
    for i in range(f1_pos + 1):
        output_preds_thresh.append({"title": output_preds[i][2],
                                    "h_idx": output_preds[i][3],
                                    "t_idx": output_preds[i][4],
                                    "r": output_preds[i][5],
                                    "evidence": []
                                    })
    time_end = time.time()
    result = "[evaluation] loss: %f, f1: %f, thresh: %f, precision: %f, recall: %f, elapsed time: %f s" \
             % (loss_avg, f1, thresh, pr_y[f1_pos], pr_x[f1_pos], time_end - time_begin)
    return result, output_preds_thresh


def predict(exe,
             program,
             pyreader,
             graph_vars,
             test_examples,
             predicate_map,
             predict_thresh):
    fetch_list = [graph_vars["logits"].name, graph_vars["ent_masks"].name]
    time_begin = time.time()
    pyreader.start()
    logits_all = None
    ent_masks_all = None
    while True:
        try:
            logits, ent_masks = exe.run(program=program, fetch_list=fetch_list)
            if logits_all is None:
                logits_all = logits
                ent_masks_all = ent_masks
            else:
                logits_all = np.append(logits_all, logits, axis=0)
                ent_masks_all = np.append(ent_masks_all, ent_masks, axis=0)

        except fluid.core.EOFException:
            pyreader.reset()
            break
    output_preds = []
    for (i, (example, logit, ent_mask)) in enumerate(zip(test_examples, logits_all, ent_masks_all)):
        for h in range(len(example.vertexSet)):
            for t in range(len(example.vertexSet)):
                if h == t:
                    continue
                if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                    continue
                for predicate_id, logit_tmp in enumerate(logit[h][t]):
                    if predicate_id == 0:
                        continue
                    else:
                        output_preds.append((logit_tmp, example.title, h, t, predicate_map[predicate_id]))
    output_preds.sort(key=lambda x: x[0], reverse=True)
    output_preds_thresh = []
    for i in range(len(output_preds)):
        if output_preds[i][0] < predict_thresh:
            break
        output_preds_thresh.append({"title": output_preds[i][1],
                                    "h_idx": output_preds[i][2],
                                    "t_idx": output_preds[i][3],
                                    "r": output_preds[i][4],
                                    "evidence": []
                                    })
    time_end = time.time()
    result = "[predict] thresh: %f, elapsed time: %f s" \
             % (predict_thresh, time_end - time_begin)
    return result, output_preds_thresh

