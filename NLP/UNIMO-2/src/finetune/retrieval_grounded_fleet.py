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
File: retrieval_grounded_fleet.py
Author: liwei(liwei85@baidu.com)
Date: 2021-10-25 10:50
Desc: Model for retrieval.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import codecs
import numpy as np

import paddle
import paddle.fluid as fluid
from finetune import img_eval
from collections import OrderedDict
from utils.utils import print_eval_log
from model.unimo_grounded import VlModel as GroundedModel
from model.unimo_grounded_baseline import VlModel as BaselineModel


def circle_loss(sp, sn, m, scale):
    """
    sp: score list of positive samples, shape [B * L]
    sn: score list of negative samples, shape [B * K]
    m: relaxation factor in circle loss function
    scale:  scale factor in circle loss function

    return: circle loss value, shape [1]
    """
    op = 1. + m
    on = 0. - m

    delta_p = 1 - m
    delta_n = m

    ap = fluid.layers.relu(op - sp)
    ap.stop_gradient = True
    an = fluid.layers.relu(sn - on)
    an.stop_gradient = True

    logit_p = ap * (sp - delta_p)
    logit_p = -1. * scale * logit_p
    logit_p = fluid.layers.cast(x=logit_p, dtype=np.float64)
    loss_p = fluid.layers.reduce_sum(fluid.layers.exp(logit_p), dim=1, keep_dim=False)

    logit_n = an * (sn - delta_n)
    logit_n = scale * logit_n
    logit_n = fluid.layers.cast(x=logit_n, dtype=np.float64)
    loss_n = fluid.layers.reduce_sum(fluid.layers.exp(logit_n), dim=1, keep_dim=False)

    circle_loss = fluid.layers.log(1 + loss_n * loss_p)
    circle_loss = fluid.layers.cast(x=circle_loss, dtype=np.float32)
    return fluid.layers.mean(circle_loss)


def create_model(args, phase, vl_config, samples_num):
    """"create_model"""
    emb_size = vl_config['hidden_size']
    patch_seq_len = vl_config['image_size'] * vl_config['image_size'] // \
                    (vl_config['resolution'] * vl_config['resolution'])

    src_ids = fluid.layers.data(name='src_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    pos_ids = fluid.layers.data(name='pos_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    sent_ids = fluid.layers.data(name='sent_ids', shape=[-1, args.max_seq_len, 1], dtype='int64')
    text_mask = fluid.layers.data(name='text_mask', shape=[-1, 1, args.max_seq_len], dtype='float32')
    image_pixel_input = fluid.layers.data(name='image_pixel_input',
                                          shape=[-1, vl_config['image_size'], vl_config['image_size'], 3],
                                          dtype='float32')
    image_mask = fluid.layers.data(name='image_mask', shape=[-1, 1, patch_seq_len + 1], dtype='float32')
    labels = fluid.layers.data(name='labels', shape=[-1, 1], dtype='int64')
    ids = fluid.layers.data(name='ids', shape=[-1, 2], dtype='int64')

    drop_last = True if phase == 'train' else False
    feed_list = [src_ids, pos_ids, sent_ids, text_mask, image_pixel_input, image_mask, labels, ids]
    pyreader = fluid.io.DataLoader.from_generator(
        feed_list=feed_list,
        capacity=70,
        use_double_buffer=True,
        iterable=False,
        drop_last=drop_last)

    text_input = {"text.word_embedding": src_ids, "text.sent_embedding": sent_ids, "text.pos_embedding": pos_ids}
    image_input = {"pixel_embedding": image_pixel_input}

    text_enc_layers = [int(i) for i in args.text_enc_layers.split(',')]
    grounding_enc_layers = [int(i) for i in args.grounding_enc_layers.split(',')]

    if args.model_type == 'grounded':
        model = GroundedModel(
            text_input=text_input,
            text_mask=text_mask,
            image_input=image_input,
            image_mask=image_mask,
            config=vl_config,
            weight_sharing=args.weight_sharing,
            grounding_method=args.grounding_method,
            topk_value=args.topk_value,
            with_grounding_projection=args.with_grounding_projection,
            with_cmcl_projection=args.with_cmcl_projection,
            cmcl_share_parameters=args.cmcl_share_parameters,
            with_grounding_pos=args.with_grounding_pos,
            text_enc_layers=text_enc_layers,
            grounding_enc_layers=grounding_enc_layers)

        if args.with_cmcl:
            cmcl_score = model.get_cmcl_scores()

    elif args.model_type == 'baseline':
        model = BaselineModel(
            text_input=text_input,
            text_mask=text_mask,
            image_input=image_input,
            image_mask=image_mask,
            config=vl_config,
            weight_sharing=args.weight_sharing,
            with_cmcl_projection=args.with_cmcl_projection,
            text_enc_layers=text_enc_layers,
            grounding_enc_layers=grounding_enc_layers)

        if args.with_cmcl:
            cmcl_score = model.get_cmcl_scores()

    else:
        raise ValueError("The model_type is invalid!!!")

    text, image = model.get_pooled_output()
    score = model.get_match_output(text, image, mode="mul")
    # if args.zero_shot:
    logits = fluid.layers.fc(
        input=score,
        size=2,
        param_attr=fluid.ParamAttr(
            name="grounded.img_text_matching_fc.w_0",
            initializer=fluid.initializer.Xavier()),
        bias_attr=fluid.ParamAttr(
            name="grounded.img_text_matching_fc.b_0",
            initializer=fluid.initializer.Constant(0.)))
    itm_score = fluid.layers.slice(logits, axes=[1], starts=[1], ends=[2])

    if args.with_cmcl:
        score = itm_score + args.cmcl_score_weight * cmcl_score
    else:
        score = itm_score

    score = fluid.layers.reshape(score, [-1, samples_num])
    if phase == 'train':
        if args.use_sigmoid:
            score = fluid.layers.sigmoid(score)
        positive_score = score[:, 0]
        image_neg_score = score[:, 1:int((samples_num + 1) / 2)]
        caption_neg_score = score[:, int((samples_num + 1) / 2):]
        acc = fluid.layers.accuracy(score, labels, k=1)

        positive_score = fluid.layers.reshape(x=positive_score, shape=[-1, 1])
        loss_c = circle_loss(positive_score, caption_neg_score, args.margin, args.scale_circle)
        loss_i = circle_loss(positive_score, image_neg_score, args.margin, args.scale_circle)
        total_loss = (loss_c + loss_i) / 2
    else:
        assert samples_num == 1
        total_loss = fluid.layers.cross_entropy(input=score, label=labels)
        total_loss = fluid.layers.mean(x=total_loss)
        acc = fluid.layers.zeros_like(total_loss)
    graph_vars = {"loss": total_loss, "acc": acc, "score": score, "label": labels, "ids": ids}
    return pyreader, graph_vars, model.get_checkpoints()


def evaluate(args, exe, test_program, test_pyreader, graph_vars, eval_phase, dev_count=1, gpu_id=0, data_reader=None):
    """evaluate"""
    test_pyreader.start()
    time_begin = time.time()
    all_mat = None
    fetch_list = [graph_vars["score"].name, graph_vars["ids"].name]
    while True:
        try:
            score, ids = exe.run(fetch_list=fetch_list)
            mat = np.concatenate([score, ids], axis=1)
            if all_mat is None:
                all_mat = mat
            else:
                all_mat = np.concatenate([all_mat, mat], axis=0)
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()

    save_file = "%s/%s.trainers_%d.part_%d.npy" % (args.eval_dir, eval_phase, dev_count, gpu_id)
    np.save(save_file, all_mat)
    tmp_file = "%s/%s.trainers_%d.part_%d.finish" % (args.eval_dir, eval_phase, dev_count, gpu_id)
    tmp_writer = codecs.open(tmp_file, "w", 'utf-8')
    tmp_writer.close()

    if gpu_id == 0:
        while True:
            ret = os.popen('find %s -maxdepth 1 -name "%s.trainers_%d.part_*.finish"' %
                           (args.eval_dir, eval_phase, dev_count)).readlines()
            if len(ret) != dev_count:
                time.sleep(1)
                continue
            else:
                break

        all_mat = None
        save_files = glob.glob("%s/%s.trainers_%d.part_*.npy" % (args.eval_dir, eval_phase, dev_count))
        for cur_save_file in save_files:
            mat = np.load(cur_save_file)
            if all_mat is None:
                all_mat = mat
            else:
                all_mat = np.concatenate([all_mat, mat], axis=0)

        cur_time = str(int(time.time()))
        os.system("mkdir %s/%s" % (args.eval_dir, cur_time))
        os.system("mv %s/%s.trainers_%d.* %s/%s" % (args.eval_dir, eval_phase, dev_count, args.eval_dir, cur_time))

        assert data_reader is not None
        text2img = {text_id: item[-1] for text_id, item in data_reader._caption_ids_dict.items()}
        img2texts = data_reader._image_sent_map

        ret = OrderedDict()
        ret['phase'] = eval_phase
        ret['loss'] = -1
        ret['data_num'] = all_mat.shape[0]
        ret['used_time'] = round(time_end - time_begin, 4)
        metrics = OrderedDict()
        metrics["recall@k"] = img_eval.recall_at_k
        if args.eval_mertrics in metrics:
            ret_metric = metrics[args.eval_mertrics](all_mat, text2img, img2texts)
            ret.update(ret_metric)
            print_eval_log(ret)
        else:
            raise ValueError('unsupported metric {}'.format(args.eval_mertrics))
        return ret
    else:
        return None
