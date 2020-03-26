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

"""rnn decoder cell
"""

import sys
import os
import traceback
import logging

import numpy as np
from paddle import fluid
from paddle.fluid import layers

from text2sql.utils import nn_utils
from text2sql import models

class RNNDecodeCell(layers.RNNCell):

    """LSTM Decoder Cell"""

    def __init__(self, hidden_size, dropout=0., init_scale=-1, name="rnn_decode_cell"):
        """init of class

        Args:
            hidden_size (TYPE): NULL
            dropout (TYPE): Default is 0.
            init_scale (TYPE): Default is -1, means paddle default initializer is used.
            name (str): param name scope

        """
        super(RNNDecodeCell, self).__init__()

        self._hidden_size = hidden_size
        self._dropout = dropout
        self._init_scale = init_scale
        self._name = name

        param = fluid.ParamAttr(initializer=nn_utils.uniform(self._init_scale))
        bias = fluid.ParamAttr(initializer=nn_utils.zero)
        self.rnn_cell = layers.LSTMCell(hidden_size, param, bias, name=name)

    def call(self, step_input, cell_state, attn_k, attn_v, padding_mask):
        """one step call

        Args:
            step_input (Variable): [batch_size, hidden_size]
            cell_state (tuple): (Variable, Variable)

        Returns: tuple
            same as input: (Variable, (Variable, Variable))

        Raises: NULL
        """
        step_feed, step_state = cell_state
        step_input = layers.concat([step_input, step_feed], 1)
        step_out, new_state = self.rnn_cell(step_input, step_state)

        decode_attn = models.Attention('dot_prod', name=self._name + '_attn')
        attn_out = decode_attn.forward(step_out, attn_k, attn_v, padding_mask=padding_mask)
        output = layers.fc(layers.concat([step_out, attn_out], axis=-1),
                           size=self._hidden_size, num_flatten_dims=1, act='tanh',
                           name=self._name + '_out_fc',
                           **nn_utils.param_attr(self._name + '_out_fc', self._init_scale, need_bias=False))
        if self._dropout > 0.:
            output = layers.dropout(x=output,
                                    dropout_prob=self._dropout,
                                    dropout_implementation="upscale_in_train")

        return output, [output, new_state]


if __name__ == "__main__":
    """run some simple test cases"""
    from text2sql.utils.debug import executor

    batch_size = 3
    hidden_size = 10

    cell = RNNDecodeCell(hidden_size)

    inputs = fluid.layers.data(name='inputs', shape=[-1, hidden_size], dtype='float32')
    states = fluid.layers.data(name='states', shape=[-1, hidden_size], dtype='float32')

    out, new_states = cell(inputs, (inputs, states))

    _data = {
            "inputs": np.random.normal(size=[batch_size, hidden_size]).astype(np.float32),
            "states": np.random.normal(size=[batch_size, hidden_size]).astype(np.float32),
        }

    exe = executor.Executor()
    result = exe.run(feed=_data, fetch_list=[out, new_states[0], new_states[1]])
    for var in result:
        print(var)
    print(result[0] == result[1])

