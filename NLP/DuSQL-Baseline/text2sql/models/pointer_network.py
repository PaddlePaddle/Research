#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""Pointer Network
"""

import sys
import os
import traceback
import logging

import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.layers import tensor

from text2sql.utils import nn_utils

INF = 1e9

class PointerNetwork(object):

    """Pointer Network"""

    def __init__(self, score_type='dot_prod', name=None, init_scale=0.1, hidden_size=-1):
        """init of class

        Args:
            score_type (TYPE): dot_prod/affine/std
            name (str): param name prefix. used for parameter sharing. Default is None.
            init_scale (float): for init fc param. used when score_type is affine or std. default is 0.1
            hidden_size (int): only used when score_type=std.
        """
        super(PointerNetwork, self).__init__()

        self._score_type = score_type
        self._name = name
        self._hidden_size = hidden_size
        self._init_scale = init_scale
        
    def forward(self, q, v, mask=None):
        """forward

        Args:
            q (Variable): shape = [batch_size, seq_len1, hidden_size] or [batch_size, hidden_size].
                          dtype = float32
            v (Variable): shape = [batch_size, seq_len2, hidden_size]. dtype = float32
            mask (Variable): shape = [batch_size, seq_len2]. dtype = v.dtype. Default is None

        Returns: Variable
            shape = [batch_size, seq_len2], dtype = float32.

        Raises:
            RuntimeError: while giving unsupported score_type.
        """
        input_dim = len(q.shape)
        if input_dim == 2:
            q = layers.unsqueeze(q, [1])

        if self._score_type == 'dot_prod':
            ptr_score = layers.matmul(q, v, transpose_y=True)
        elif self._score_type == 'affine':
            q_tmp = layers.fc(q, size=v.shape[2], num_flatten_dims=2,
                              **nn_utils.param_attr(self._name, self._init_scale, need_bias=True))
            ptr_score = layers.matmul(q_tmp, v, transpose_y=True)
        elif self._score_type == 'std':
            if self._hidden_size <= 0:
                raise ValueError("hidden_size should greater than 0")
            q_tmp = layers.fc(q, size=self._hidden_size, num_flatten_dims=2,
                              **nn_utils.param_attr(self._name + '_q', self._init_scale, need_bias=True))
            v_tmp = layers.fc(v, size=self._hidden_size, num_flatten_dims=2,
                              **nn_utils.param_attr(self._name + '_k', self._init_scale, need_bias=True))

            # shape = [batch_size, seq_len1, seq_len2, hidden_size]
            q_tmp_expand = layers.expand(layers.unsqueeze(q_tmp, [2]), [1, 1, v_tmp.shape[1], 1])
            # shape = [batch_size, 1, seq_len2, hidden_size]
            v_tmp_expand = layers.unsqueeze(v_tmp, [1])
            ptr_score = layers.fc(layers.elementwise_add(q_tmp_expand, v_tmp_expand, act='tanh'),
                                  size=1,
                                  num_flatten_dims=3,
                                  **nn_utils.param_attr(self._name + '_w', self._init_scale, need_bias=True))
            ptr_score = layers.squeeze(ptr_score, [3])
        else:
            raise RuntimeError('Supported score types: dot_prod/affine/std. but got %s' % (self._score_type))

        if mask is not None:
            score_for_mask = layers.transpose(ptr_score, [1, 0, 2])
            ptr_score_masked = layers.elementwise_add(score_for_mask, (mask - 1.0) * INF, axis=-1)
            ptr_score = layers.transpose(ptr_score_masked, [1, 0, 2])

        if input_dim == 2:
            ptr_score = layers.squeeze(ptr_score, [1])
        return ptr_score


if __name__ == "__main__":
    """run some simple test cases"""
    from text2sql.utils.debug import executor

    batch_size = 2
    hidden_size = 3

    q = fluid.layers.data(name='q', shape=[-1, 3, hidden_size], dtype='float32')
    v = fluid.layers.data(name='v', shape=[-1, 4, hidden_size], dtype='float32')
    lens = fluid.layers.data(name='lens', shape=[-1, 4], dtype='float32')
    mask = layers.sequence_mask(lens, maxlen=4, dtype='float32')

    ptr_net = PointerNetwork('dot_prod', name='dot')
    weight_dot = ptr_net.forward(q, v, mask=mask)
    ptr_net = PointerNetwork('affine', name='dot')
    weight_aff = ptr_net.forward(q, v, mask=mask)
    ptr_net = PointerNetwork('std', hidden_size=100, name='dot')
    weight_std = ptr_net.forward(q, v, mask=mask)

    def _data():
        return {
            "q": np.random.normal(size=[batch_size, 3, hidden_size]).astype(np.float32),
            "v": np.random.normal(size=[batch_size, 4, hidden_size]).astype(np.float32),
            "lens": np.array([2, 3]).astype(np.int32),
        }

    exe = executor.Executor()
    result = exe.run(feed=_data(), fetch_list=[weight_dot, weight_aff, weight_std, mask])
    for var in result:
        print(var)




