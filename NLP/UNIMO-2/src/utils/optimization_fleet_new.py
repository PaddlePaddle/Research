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
File: optimization_fleet_new.py
Author: liwei(liwei85@baidu.com)
Date: 2021-06-02 20:16
Desc: Optimization and learning rate scheduling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.distributed.fleet as fleet


def optimization(loss,
                 warmup_steps,
                 num_train_steps,
                 learning_rate,
                 train_program,
                 weight_decay,
                 scheduler='linear_warmup_decay',
                 beta1=0.9,
                 beta2=0.98,
                 epsilon=1e-06,
                 boundaries=None,
                 values=None,
                 dist_strategy=None):

    if warmup_steps > 0:
        if scheduler == 'noam_decay':
            scheduled_lr = paddle.optimizer.lr.NoamDecay(
                1 / (warmup_steps * (learning_rate ** 2)), warmup_steps)
        elif scheduler == 'linear_warmup_decay':
            scheduled_lr = paddle.optimizer.lr.LinearWarmup(
                learning_rate=paddle.optimizer.lr.PolynomialDecay(learning_rate=learning_rate,
                                                                  decay_steps=num_train_steps,
                                                                  end_lr=0.0,
                                                                  power=1.0,
                                                                  cycle=False),
                warmup_steps=warmup_steps,
                start_lr=0,
                end_lr=learning_rate)
        elif scheduler == 'scale_by_epoch_decay':
            if boundaries is None:
                boundaries = [10000, 20000]
            if values is None:
                values = [5e-6, 5e-7, 5e-8]
            scheduled_lr = paddle.optimizer.lr.PiecewiseDecay(boundaries=boundaries, values=values)
        else:
            raise ValueError("Unkown learning rate scheduler, should be "
                             "'noam_decay' or 'linear_warmup_decay'")
    else:
        scheduled_lr = paddle.optimizer.lr.LRScheduler(learning_rate=learning_rate)

    # FIXME: added here
    def exclude_from_weight_decay(name):
        name = name.rstrip('.master')
        if name.find("layer_norm") > -1:
            return False
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return False
        return True

    optimizer = paddle.optimizer.AdamW(learning_rate=scheduled_lr,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon,
                                       grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
                                       weight_decay=weight_decay,
                                       apply_decay_param_fun=exclude_from_weight_decay)

    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)
    optimizer.minimize(loss)
    return scheduled_lr
