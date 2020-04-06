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

"""Attention
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

class Attention(object):

    """Attention"""

    def __init__(self, score_type, name=None, init_scale=0.1, hidden_size=-1):
        """init of class

        Args:
            score_type (TYPE): dot_prod/affine/std
            name (str): param name prefix. used for parameter sharing. Default is None.
        """
        super(Attention, self).__init__()

        self._score_type = score_type
        self._name = name
        self._hidden_size = hidden_size
        self._init_scale = init_scale
        
    def forward(self, q, k, v=None, padding_mask=None, num_heads=1):
        """forward

        Args:
            q (Variable): shape = [batch_size, seq_len1, hidden_size_q]
            k (Variable): shape = [batch_size, seq_len2, hidden_size_k]
            v (Variable): shape = [batch_size, seq_len2, hidden_size_v]
            mask (Variable): lens of k and v. Default is None
            num_heads (int): currently only support 1. Default is 1

        Returns: TODO

        Raises: NULL
        """
        q_shape = q.shape
        if len(q_shape) == 2:
            q = layers.unsqueeze(q, [1])

        if v is None:
            v = k
        if self._score_type == 'dot_prod':
            # [batch_size, q_lens, k_lens]
            attn_score = layers.matmul(q, k, transpose_y=True)
        elif self._score_type == 'affine':
            k_tmp = layers.fc(k, size=q.shape[2], num_flatten_dims=2,
                              **nn_utils.param_attr(self._name, self._init_scale, need_bias=True))
            attn_score = layers.matmul(q, k_tmp, transpose_y=True)
        elif self._score_type == 'std':
            if self._hidden_size <= 0:
                raise ValueError("hidden_size should greater than 0")
            q_tmp = layers.fc(q, size=self._hidden_size, num_flatten_dims=2,
                              **nn_utils.param_attr(self._name + '_q', self._init_scale, need_bias=True))
            k_tmp = layers.fc(k, size=self._hidden_size, num_flatten_dims=2,
                              **nn_utils.param_attr(self._name + '_k', self._init_scale, need_bias=True))

            # shape = [batch_size, seq_len1, seq_len2, hidden_size]
            q_tmp_expand = layers.expand(layers.unsqueeze(q_tmp, [2]), [1, 1, v.shape[1], 1])
            # shape = [batch_size, 1, seq_len2, hidden_size]
            k_tmp_expand = layers.unsqueeze(k_tmp, [1])
            attn_score = layers.fc(layers.elementwise_add(q_tmp_expand, k_tmp_expand, act='tanh'),
                                  size=1,
                                  num_flatten_dims=3,
                                  **nn_utils.param_attr(self._name + '_w', self._init_scale, need_bias=True))
            attn_score = layers.squeeze(attn_score, [3])
        else:
            raise RuntimeError('Supported score types: dot_prod/affine/std. but got %s' % (self._score_type))

        if padding_mask is not None:
            attn_for_mask = layers.transpose(attn_score, [1, 0, 2])
            attn_score_masked = layers.elementwise_add(attn_for_mask, padding_mask * INF, axis=-1)
            attn_score = layers.transpose(attn_score_masked, [1, 0, 2])

        weight = layers.softmax(attn_score)
        attn = layers.matmul(weight, v)
        if len(q_shape) == 2:
            attn = layers.squeeze(attn, [1])

        return attn


if __name__ == "__main__":
    """run some simple test cases"""
    from text2sql.utils.debug import executor

    batch_size = 2
    hidden_size = 3

    q = fluid.layers.data(name='q', shape=[-1, 3, hidden_size], dtype='float32')
    k = fluid.layers.data(name='k', shape=[-1, 4, hidden_size], dtype='float32')
    lens = fluid.layers.data(name='lens', shape=[-1, 4], dtype='float32')
    mask = layers.sequence_mask(lens, maxlen=4, dtype='float32')

    attn = Attention('dot_prod', name='dot')
    ctx_dot = attn.forward(q, k, k, mask=mask)
    attn = Attention('affine', name='dot')
    ctx_aff = attn.forward(q, k, k, mask=mask)
    attn = Attention('std', hidden_size=100, name='dot')
    ctx_std = attn.forward(q, k, k, mask=mask)

    def _data():
        return {
            "q": np.random.normal(size=[batch_size, 3, hidden_size]).astype(np.float32),
            "k": np.random.normal(size=[batch_size, 4, hidden_size]).astype(np.float32),
            "lens": np.array([2, 3]).astype(np.int32),
        }

    exe = executor.Executor()
    result = exe.run(feed=_data(), fetch_list=[ctx_dot, ctx_aff, ctx_std, mask])
    for var in result:
        print(var)



