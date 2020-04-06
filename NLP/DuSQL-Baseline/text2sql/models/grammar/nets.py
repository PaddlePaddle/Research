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

"""grammar networks
"""

import sys
import os
import traceback
import logging
from collections import namedtuple

import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.layers import tensor

from text2sql import models
from text2sql.utils import fluider
from text2sql.utils import nn_utils

INIT_SCALE = -1
INF = 1e9

def _apply_rule(condition, inputs, gmr_mask, grammar, name=None):
    """apply_rule.

    Args:
        condition (TYPE): NULL
        inputs (Variable): shape = [batch_size, max_len, hidden_size]. infer 阶段 max_len 恒为1
        gmr_mask (TYPE): NULL
        grammar (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    fc_name = None
    if name is not None:
        fc_name = name + '_apply_rule_fc'

    condition = layers.cast(condition, dtype='float32')
    gmr_output = layers.fc(inputs, size=grammar.grammar_size,
                           **nn_utils.param_attr(fc_name, INIT_SCALE, need_bias=True))
    gmr_output_masked = layers.elementwise_add(gmr_output, gmr_mask)

    zeros = layers.fill_constant_batch_size_like(
                                gmr_output_masked,
                                shape=[-1, grammar.MAX_TABLE + grammar.MAX_COLUMN + grammar.MAX_VALUE],
                                dtype='float32',
                                value=-INF)
    final_output = tensor.concat([gmr_output_masked, zeros], axis=-1)
    true_final_output = layers.elementwise_mul(final_output, condition, axis=0)
    return true_final_output


def _select_table(condition, inputs, table_enc, table_len, table_mask_by_col, ptr_net, grammar, name=None):
    """select_table.

    Args:
        condition (TYPE): NULL
        inputs (Variable): shape = [batch_size, max_len, hidden_size]. infer 阶段 max_len 恒为1
        table_enc (TYPE): NULL
        table_len (TYPE): NULL
        ptr_net (TYPE): NULL
        grammar (TYPE): NULL
        name (str):
        table_mask_by_col (Variable):

    Returns: TODO

    Raises: NULL
    """
    condition = layers.cast(condition, dtype='float32')

    table_mask_by_len = layers.sequence_mask(table_len, maxlen=grammar.MAX_TABLE, dtype='float32')
    table_mask_by_len = layers.reshape(table_mask_by_len, [-1, grammar.MAX_TABLE])
    table_mask_by_col = layers.reshape(table_mask_by_col, [-1, grammar.MAX_TABLE])
    table_mask = layers.elementwise_mul(table_mask_by_len, table_mask_by_col)
    predicts = ptr_net.forward(inputs, table_enc, table_mask)

    zeros_l = tensor.fill_constant_batch_size_like(predicts,
                            shape=[-1, grammar.grammar_size], dtype='float32', value=-INF)
    zeros_r = tensor.fill_constant_batch_size_like(predicts,
                            shape=[-1, grammar.MAX_COLUMN + grammar.MAX_VALUE], dtype='float32', value=-INF)
    final_output = tensor.concat([zeros_l, predicts, zeros_r], axis=-1)
    true_final_output = layers.elementwise_mul(final_output, condition, axis=0)
    return true_final_output


def _select_column(condition, inputs, column_enc, column_len, ptr_net, grammar, column2table_mask, name=None):
    """select_column.

    Args:
        condition (TYPE): NULL
        inputs (Variable): shape = [batch_size, max_len, hidden_size]. infer 阶段 max_len 恒为1
        column_enc (TYPE): NULL
        column_len (TYPE): NULL
        ptr_net (TYPE): NULL
        grammar (TYPE): NULL
        column2table_mask (Variable):
        name (str):

    Returns: TODO

    Raises: NULL
    """
    condition = layers.cast(condition, dtype='float32')

    column_mask = layers.sequence_mask(column_len, maxlen=grammar.MAX_COLUMN, dtype='float32')
    column_mask = layers.reshape(column_mask, [-1, grammar.MAX_COLUMN])
    predicts = ptr_net.forward(inputs, column_enc, column_mask)

    pred_ids = layers.argmax(predicts, axis=-1)
    valid_table_mask = nn_utils.batch_gather(column2table_mask, pred_ids)

    ## concat zeros to vocab size
    zeros_l = tensor.fill_constant_batch_size_like(predicts,
                            shape=[-1, grammar.grammar_size + grammar.MAX_TABLE], dtype='float32', value=-INF)
    zeros_r = tensor.fill_constant_batch_size_like(predicts,
                            shape=[-1, grammar.MAX_VALUE], dtype='float32', value=-INF)
    final_output = tensor.concat([zeros_l, predicts, zeros_r], axis=-1)
    true_final_output = layers.elementwise_mul(final_output, condition, axis=0)
    true_valid_table_mask = layers.elementwise_mul(valid_table_mask, condition, axis=0)
    return true_final_output, true_valid_table_mask


def _select_value(condition, inputs, value_enc, value_len, ptr_net, grammar, name=None):
    """select_value.

    Args:
        condition (TYPE): NULL
        inputs (TYPE): NULL
        value_enc (TYPE): NULL
        value_len (TYPE): NULL
        ptr_net (TYPE): NULL
        grammar (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    condition = layers.cast(condition, dtype='float32')

    value_mask = layers.sequence_mask(value_len, maxlen=grammar.MAX_VALUE, dtype='float32')
    value_mask = layers.reshape(value_mask, [-1, grammar.MAX_VALUE])
    predicts = ptr_net.forward(inputs, value_enc, value_mask)

    ## concat zeros to vocab size
    zeros_l = tensor.fill_constant_batch_size_like(predicts,
                        shape=[-1, grammar.grammar_size + grammar.MAX_TABLE + grammar.MAX_COLUMN],
                        dtype='float32', value=-INF)
    final_output = tensor.concat([zeros_l, predicts], axis=-1)
    true_final_output = layers.elementwise_mul(final_output, condition, axis=0)
    return true_final_output


def grammar_output(inputs, actions, gmr_mask, last_col2tbl_mask, decode_vocab, grammar, name=None, column2table=None):
    """output logits according to grammar

    Args:
        inputs (Variable): shape = [batch_size, max_len, hidden_size]. infer 阶段 max_len 恒为1
        actions (Variable): shape = [batch_size, max_len]. infer 阶段 max_len 恒为1
        gmr_mask (Variable): shape = [batch_size, max_len, grammar_size]. infer 阶段 max_len 恒为1
        last_col2tbl_mask (Variable): shape = [batch_size, max_len, max_table]. 解码过程中，上一个step为column时，其对应的 table mask
        decode_vocab (DecoderDynamicVocab): (table, table_len, column, column_len, value, value_len, column2table_mask).
                                            这里的column2table_mask是跟column一一对应的table mask。
        gramamr (Grammar): NULL
        name (str): Variable 的 name 前缀。用于多次调用时的参数共享。默认为 None，表示参数不会共享。

    Returns: (Variable, Variable)
        output: 词表输出概率
        valid_table_mask: 只在预测阶段有效

    Raises: NULL
    """
    batch_size = layers.shape(inputs)[0]
    max_len = inputs.shape[1]
    vocab_size = grammar.vocab_size

    action_shape = [batch_size, max_len]
    act_apply_rule = tensor.fill_constant(shape=action_shape, value=grammar.ACTION_APPLY, dtype='int64')
    act_stop = tensor.fill_constant(shape=action_shape, value=grammar.ACTION_STOP, dtype='int64')
    act_select_t = tensor.fill_constant(shape=action_shape, value=grammar.ACTION_SELECT_T, dtype='int64')
    act_select_c = tensor.fill_constant(shape=action_shape, value=grammar.ACTION_SELECT_C, dtype='int64')
    act_select_v = tensor.fill_constant(shape=action_shape, value=grammar.ACTION_SELECT_V, dtype='int64')
    cond_apply_rule = layers.logical_or(layers.equal(actions, act_apply_rule),
                                        layers.equal(actions, act_stop))
    cond_select_t = layers.equal(actions, act_select_t)
    cond_select_c = layers.equal(actions, act_select_c)
    cond_select_v = layers.equal(actions, act_select_v)

    # expand vocab to [-1, max_len, ...]
    if max_len == 1:
        expand_to_seq_len = lambda x: layers.unsqueeze(x, [1])
    else:
        expand_to_seq_len = lambda x: layers.expand(layers.unsqueeze(x, [1]), [1, max_len] + [1] * (len(x.shape) - 1))
    table_enc = expand_to_seq_len(decode_vocab.table)
    table_len = expand_to_seq_len(decode_vocab.table_len)
    column_enc = expand_to_seq_len(decode_vocab.column)
    column_len = expand_to_seq_len(decode_vocab.column_len)
    value_enc = expand_to_seq_len(decode_vocab.value)
    value_len = expand_to_seq_len(decode_vocab.value_len)
    column2table_mask = expand_to_seq_len(decode_vocab.column2table_mask)

    # merge batch & seq_len dim
    inputs = nn_utils.merge_first_ndim(inputs, n=2)
    actions = nn_utils.merge_first_ndim(actions, n=2)
    gmr_mask = nn_utils.merge_first_ndim(gmr_mask, n=2)
    last_col2tbl_mask = nn_utils.merge_first_ndim(last_col2tbl_mask, n=2)
    table_enc = nn_utils.merge_first_ndim(table_enc, n=2)
    table_len = nn_utils.merge_first_ndim(table_len, n=2)
    column_enc = nn_utils.merge_first_ndim(column_enc, n=2)
    column_len = nn_utils.merge_first_ndim(column_len, n=2)
    value_enc = nn_utils.merge_first_ndim(value_enc, n=2)
    value_len = nn_utils.merge_first_ndim(value_len, n=2)
    column2table_mask = nn_utils.merge_first_ndim(column2table_mask, n=2)
    cond_apply_rule = nn_utils.merge_first_ndim(cond_apply_rule, n=2)
    cond_select_t = nn_utils.merge_first_ndim(cond_select_t, n=2)
    cond_select_c = nn_utils.merge_first_ndim(cond_select_c, n=2)
    cond_select_v = nn_utils.merge_first_ndim(cond_select_v, n=2)

    t_ptr_net = models.PointerNetwork(score_type="affine", name='gmr_output_t_ptr')
    c_ptr_net = models.PointerNetwork(score_type="affine", name='gmr_output_c_ptr')
    v_ptr_net = models.PointerNetwork(score_type="affine", name='gmr_output_v_ptr')

    ## 核心处理逻辑 ##
    apply_rule_output = _apply_rule(cond_apply_rule, inputs, gmr_mask, grammar, name=name)
    select_t_output = \
            _select_table(cond_select_t, inputs, table_enc, table_len, last_col2tbl_mask, t_ptr_net, grammar)
    select_c_output, valid_table_mask = \
            _select_column(cond_select_c, inputs, column_enc, column_len, c_ptr_net, grammar, column2table_mask)
    select_v_output = _select_value(cond_select_v, inputs, value_enc, value_len, v_ptr_net, grammar)

    output = fluider.elementwise_add(apply_rule_output, select_t_output, select_c_output, select_v_output, axis=0)
    output = layers.reshape(output, shape=[batch_size, max_len, vocab_size])
    return output, valid_table_mask


if __name__ == "__main__":
    """run some simple test cases"""
    from text2sql.utils.debug import executor
    from text2sql.grammar import Grammar
    from text2sql.models.grammar import DecoderDynamicVocab

    Grammar.MAX_TABLE = 12
    grammar = Grammar('conf/grammar.txt')

    batch_size = 2
    seq_len = 4
    hidden_size = 3

    inputs = layers.data(name='inputs', shape=[-1, seq_len, hidden_size], dtype='float32')
    input_mask = layers.data(name='input_mask', shape=[-1, seq_len, 1], dtype='float32')
    actions = layers.data(name='actions', shape=[-1, seq_len], dtype='int64')
    gmr_mask = layers.data(name='gmr_mask', shape=[-1, seq_len, grammar.grammar_size], dtype='float32')
    decode_vocab = DecoderDynamicVocab(
                    fluid.layers.data(name='table', shape=[-1, grammar.MAX_TABLE, hidden_size], dtype='float32'),
                    fluid.layers.data(name='table_len', shape=[-1, 1], dtype='int64'),
                    fluid.layers.data(name='column', shape=[-1, grammar.MAX_COLUMN, hidden_size], dtype='float32'),
                    fluid.layers.data(name='column_len', shape=[-1, 1], dtype='int64'),
                    fluid.layers.data(name='value', shape=[-1, grammar.MAX_VALUE, hidden_size], dtype='float32'),
                    fluid.layers.data(name='value_len', shape=[-1, 1], dtype='int64'))

    valid_table_mask = layers.zeros(shape=[batch_size * 1, grammar.MAX_TABLE], dtype='float32')
    output = grammar_output(inputs, actions, gmr_mask, valid_table_mask, decode_vocab, grammar, name='gmr_decode_out')
    output_mask = layers.elementwise_mul(output, input_mask, axis=0)
    output_id = layers.argmax(output_mask, axis=-1)

    _data = {
            "inputs": np.random.normal(size=[batch_size, seq_len, hidden_size]).astype(np.float32),
            "input_mask": np.array([[[1], [1], [1], [1]],
                                 [[1], [1], [1], [0]]]).astype(np.float32),
            "actions": np.array([[1, 1, 2, 3],
                                 [1, 4, 3, 0]]).astype(np.int64),
            "gmr_mask": np.random.normal(size=[batch_size, seq_len, grammar.grammar_size]).astype(np.float32),
            "table": np.random.normal(size=[batch_size, grammar.MAX_TABLE, hidden_size]).astype(np.float32),
            "table_len": np.array([[8], [5]]).astype(np.int64),
            "column": np.random.normal(size=[batch_size, grammar.MAX_COLUMN, hidden_size]).astype(np.float32),
            "column_len": np.array([[4], [9]]).astype(np.int64),
            "value": np.random.normal(size=[batch_size, grammar.MAX_VALUE, hidden_size]).astype(np.float32),
            "value_len": np.array([[5], [7]]).astype(np.int64),
        }

    exe = executor.Executor()
    result = exe.run(feed=_data, fetch_list=[output, output_mask, output_id])
    for var in result:
        print(var)


