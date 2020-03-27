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

"""table encoder based on rnn
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

class Sequence2DEncoder(object):

    """Sequence2DEncoder"""

    def __init__(self, enc_type, dropout=0.0, init_scale=0.1, name=None, **kwargs):
        """init of class

        Args:
            enc_type (str): simple_sum|birnn
            num_layers (int): param for rnn encoder. default is 1.
            hidden_size (int): param for rnn encoder. default is 0
            bidirectional (bool): default is False
            dropout (float): default is 0.0
            init_scale (float): default is 0.1
            name (str): default is None
            kwargs (dict): extract args for different encoder
        """
        super(Sequence2DEncoder, self).__init__()

        self._enc_type = enc_type

        if self._enc_type == 'birnn':
            self._hidden_size = kwargs['hidden_size']
            self._num_layers = kwargs.get('num_layers', 1)
            self._bidirectional = kwargs.get('bidirectional', False)
            self._dropout = dropout
            self._init_scale = init_scale

            self._rnn_encoder = models.RNNEncoder(
                    self._num_layers, self._hidden_size, self._bidirectional, self._dropout, self._init_scale, name)
            self._encoder = self._birnn_encoder
        elif self._enc_type == 'simple_sum':
            self._encoder = self._simple_sum_encoder
        else:
            raise ValueError('enc_type setting error. expect birnn|simple_sum, but got %s' % (enc_type))

    def forward(self, inputs, input_len, name_lens, name_pos, name_tok_len):
        """forward

        Args:
            inputs (Variable): shape=[batch_size, max_seq_len, hidden_size]
            input_len (Variable): shape=[batch_size]
            name_lens (Variable): shape=[batch_size]
            name_pos (Variable): shape=[batch_size, max_name_len, max_tokens]
            name_tok_len (Variable): shape=[batch_size, max_name_len]

        Returns: TODO

        Raises: NULL

        """
        return self._encoder(inputs, input_len, name_lens, name_pos, name_tok_len)

    def _birnn_encoder(self, inputs, input_len, name_lens, name_pos, name_tok_len):
        """forward

        Args:
            inputs (Variable): shape=[batch_size, max_seq_len, hidden_size]
            input_len (Variable): shape=[batch_size]
            name_lens (Variable): shape=[batch_size]
            name_pos (Variable): shape=[batch_size, max_name_len, max_tokens]
            name_tok_len (Variable): shape=[batch_size, max_name_len]

        Returns: TODO

        Raises: NULL

        """
        rnn_output, rnn_final_state = self._rnn_encoder.forward(inputs, input_len)

        max_name_len = name_pos.shape[1]
        name_begin = name_pos[:, :, 0]

        name_repr_mask = layers.sequence_mask(name_lens, max_name_len, dtype=name_tok_len.dtype)
        len_delta = layers.elementwise_mul(name_tok_len - 1, name_repr_mask, axis=0)
        name_end = name_begin + len_delta

        if self._bidirectional:
            name_fwd_repr_gathered = nn_utils.batch_gather_2d(rnn_output, name_end)[:, :, :self._hidden_size]
            name_bwd_repr_gathered = nn_utils.batch_gather_2d(rnn_output, name_begin)[:, :, self._hidden_size:]
            name_repr_gathered = layers.concat(input=[name_fwd_repr_gathered, name_bwd_repr_gathered], axis=-1)
            new_hidden_size = self._hidden_size * 2
        else:
            name_repr_gathered = layers.gather_nd(rnn_output, name_end)
            new_hidden_size = self._hidden_size

        name_repr_tmp = layers.reshape(name_repr_gathered, shape=[-1, max_name_len, new_hidden_size])
        name_repr_mask = layers.cast(name_repr_mask, dtype=name_repr_tmp.dtype)
        name_repr = layers.elementwise_mul(name_repr_tmp, name_repr_mask, axis=0)

        return name_repr, None

    def _simple_sum_encoder(self, inputs, input_len, name_lens, name_pos, name_tok_len):
        """forward

        Args:
            inputs (Variable): shape=[batch_size, max_seq_len, hidden_size]
            input_len (Variable): shape=[batch_size]
            name_lens (Variable): shape=[batch_size]
            name_pos (Variable): shape=[batch_size, max_name_len, max_tokens]
            name_tok_len (Variable): shape=[batch_size, max_name_len]

        Returns: TODO

        Raises: NULL

        """
        max_name_len = name_pos.shape[1]
        max_name_tok_len = name_pos.shape[2]
        hidden_size = inputs.shape[2]

        name_pos_1d = layers.reshape(name_pos, shape=[-1, max_name_len * max_name_tok_len])
        name_enc = nn_utils.batch_gather_2d(inputs, name_pos_1d)
        name_enc = layers.reshape(name_enc, shape=[-1, max_name_len, max_name_tok_len, hidden_size])

        # shape = [batch_size, name_len, token_len, hidden_size]
        name_tok_mask = layers.sequence_mask(name_tok_len, maxlen=max_name_tok_len, dtype=name_enc.dtype)
        name_enc_masked = layers.elementwise_mul(name_enc, name_tok_mask, axis=0)
        # shape = [batch_size, name_len, hidden_size]
        output = layers.reduce_sum(name_enc_masked, dim=2)
        return output, None


if __name__ == "__main__":
    """run some simple test cases"""
    pass

