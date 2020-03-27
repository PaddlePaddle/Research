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

"""rnn encoder
"""

import sys
import os
import traceback
import logging

import numpy as np
from paddle import fluid
from paddle.fluid import layers

from text2sql.utils import nn_utils

class BasicRNNCell(layers.RNNCell):

    """Implementation of rnn cell"""

    def __init__(self, hidden_size, cell_type="lstm", dropout=0.0, init_scale=-1, name="BasicRNNCell"):
        """init of class

        Args:
            hidden_size (TYPE): NULL
            cell_type (str): lstm|gru
            dropout (TYPE): Default is 0.0
            init_scale (TYPE): Default is -1
            name (TYPE): Default is "BasicRNNCell"

        """
        super(BasicRNNCell, self).__init__()

        self._hidden_size = hidden_size
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._init_scale = init_scale
        self._name = name

        param = fluid.ParamAttr(initializer=nn_utils.uniform(self._init_scale))
        bias = fluid.ParamAttr(initializer=nn_utils.zero)
        if self._cell_type == 'lstm':
            self._cell = layers.LSTMCell(self._hidden_size, param, bias, name=self._name)
        elif self._cell_type == 'gru':
            self._cell = layers.GRUCell(self._hidden_size, param, bias, name=self._name)
        else:
            raise ValueError("cell type only supported <lstm|gru>, but got %s" % (cell_type))

    def call(self, step_input, states):
        """call one step of rnn

        Args:
            step_input (TYPE): NULL
            states (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        step_out, new_state = self._cell(step_input, states)
        if self._dropout > 0:
            step_out = layers.dropout(step_out, self._dropout, dropout_implementation="upscale_in_train")

        return step_out, new_state

    @property
    def state_shape(self):
        """state shape"""
        return self._cell.state_shape
        

class BiRNN(object):

    """BiRNN."""

    def __init__(self, hidden_size, bidirectional=True, dropout=0.0, init_scale=-1, name="BiRNNLayer"):
        """init of class 

        Args:
            hidden_size (int): NULL
            bidirectional (bool): default is False
            dropout (float): default is 0.0
            init_scale (float): default is -1, means paddle default initializer is used.
            name (str): default is "BiRNNCell"
        """
        super(BiRNN, self).__init__()

        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._dropout = dropout
        self._init_scale = init_scale
        if name is None:
            name = 'BiRNNLayer'

        self.fwd_cell = BasicRNNCell(self._hidden_size,
                                     cell_type='lstm',
                                     dropout=self._dropout,
                                     init_scale=self._init_scale,
                                     name=name + '_fwd')
        self.bwd_cell = None
        if bidirectional:
            self.bwd_cell = BasicRNNCell(self._hidden_size,
                                         cell_type='lstm',
                                         dropout=self._dropout,
                                         init_scale=self._init_scale,
                                         name=name + '_bwd')

    def forward(self, inputs, input_lens):
        """run bi-rnn

        Args:
            inputs (TYPE): NULL
            input_lens (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        fwd_output, fwd_final_state = layers.rnn(self.fwd_cell, inputs, sequence_length=input_lens)
        if self._bidirectional:
            bwd_output, bwd_final_state = layers.rnn(self.bwd_cell, inputs, sequence_length=input_lens, is_reverse=True)

            output = layers.concat(input=[fwd_output, bwd_output], axis=-1)
            final_state = [
                layers.concat(input=[fwd_final_state[0], bwd_final_state[0]], axis=-1),
                layers.concat(input=[fwd_final_state[1], bwd_final_state[1]], axis=-1),
                ]
        else:
            output = fwd_output
            final_state = fwd_final_state

        return output, final_state
        

class RNNEncoder(object):

    """rnn encoder"""

    def __init__(self, num_layers, hidden_size, bidirectional=False, dropout=0.0, init_scale=-1, name="RNNEncoder"):
        """init of class

        Args:
            num_layers (TYPE): NULL
            hidden_size (TYPE): NULL
            bidirectional (TYPE): default is False
            dropout (TYPE): default is 0.0
            init_scale (TYPE): default is -1, means paddle default initializer is used.
            name (str): default is "RNNEncoder"

        """
        super(RNNEncoder, self).__init__()

        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._rnn_layers = []

        if name is None:
            name = 'RNNEncoder'
        for i in range(num_layers):
            curr_name = '%s_layer%d' % (name, i)
            self._rnn_layers.append(BiRNN(hidden_size, bidirectional, dropout, init_scale, name=curr_name))

    def forward(self, inputs, input_lens):
        """call of rnn encoder

        Args:
            inputs (TYPE): NULL
            input_lens (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        inputs_tmp = inputs
        for i in range(self._num_layers):
            output, final_state = self._rnn_layers[i].forward(inputs_tmp, input_lens)
            inputs_tmp = output

        return output, final_state


if __name__ == "__main__":
    """run some simple test cases"""
    pass

