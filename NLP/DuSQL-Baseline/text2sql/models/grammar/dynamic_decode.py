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

"""do decoding process with grammar constraint
"""

import sys
import os
import traceback
import logging
import functools

import numpy as np
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.layers import tensor

from text2sql.grammar import Grammar
from text2sql.utils import fluider
from text2sql.utils import nn_utils
from text2sql.utils import data_structure

__all__ = ["decode_with_grammar"]

STACK_EXPAND_TIMES = 5


def _push_to_stack(gmr_desc, gmr_pos, gmr_lens, gmr_stack_info):
    """push grammar id in gmr_desc from gmr_pos to gmr_lens to
    gmr_stack. and update step_gmr_pos

    Args:
        gmr_desc (TYPE): NULL
        gmr_pos (TYPE): NULL
        gmr_lens (TYPE): NULL
        gmr_stack_info (tuple): [in/out] (gmr_stack, gmr_stack_pos)

    Returns: tuple (gmr_stack, gmr_stack_pos)

    Raises: NULL
    """
    gmr_stack, gmr_stack_pos = gmr_stack_info
    mv_step = layers.cast(layers.greater_than(gmr_lens, layers.zeros_like(gmr_lens)), dtype=gmr_lens.dtype)
    gmr_mv_pos = layers.elementwise_sub(gmr_lens, mv_step)

    cond = layers.reduce_any(layers.greater_than(gmr_mv_pos, gmr_pos))
    while_op = layers.While(cond)
    with while_op.block():
        gmr_ids = nn_utils.batch_gather(gmr_desc, gmr_mv_pos)
        gmr_stack_tmp, gmr_stack_pos_tmp = data_structure.Stack.push(gmr_stack_info, gmr_ids, in_place=False)

        mv_cond = layers.greater_than(gmr_mv_pos, gmr_pos)
        gmr_mv_pos_tmp = fluider.elementwise_sub(gmr_mv_pos, mv_cond, force=True)
        new_gmr_stack, new_gmr_stack_pos = nn_utils.ifelse(mv_cond,
                [gmr_stack_tmp, gmr_stack_pos_tmp],
                [gmr_stack, gmr_stack_pos])
        layers.utils.map_structure(layers.assign,
                [new_gmr_stack, new_gmr_stack_pos],
                [gmr_stack, gmr_stack_pos])
        layers.assign(gmr_mv_pos_tmp, gmr_mv_pos)
        layers.assign(layers.reduce_any(layers.greater_than(gmr_mv_pos, gmr_pos)), cond)
    return gmr_stack, gmr_stack_pos


def _process_type_leaf(condition, decoder, grammar_stack, next_inputs, finished):
    """Process when output type is LEAF

    Args:
        condition (TYPE): NULL
        decoder (TYPE): NULL
        grammar_stack (StackData): (gmr_stack_data, gmr_stack_pos)
        next_inputs (DecoderInputsWrapper): (input_var, action, grammar_mask)
        finished (TYPE): NULL

    Returns: None

    Raises: NULL
    """
    ## pop stack
    next_output, valid_pos, gmr_stack_tmp = data_structure.Stack.pop(grammar_stack, mask=True, in_place=False)
    valid_pos = fluider.squeeze(valid_pos, [1])

    ## update next grammar mask
    next_actions = layers.elementwise_mul(decoder.grammar_action(next_output),
                            layers.cast(valid_pos, dtype=next_inputs.action.dtype),
                            axis=0)
    next_gmr_mask = layers.elementwise_mul(decoder.grammar_mask(next_output),
                            layers.cast(valid_pos, dtype=next_inputs.gmr_mask.dtype),
                            axis=0)

    ## save result, while condition is True
    new_gmr_stack_data, new_gmr_stack_pos, new_actions, new_gmr_mask = nn_utils.ifelse(condition,
            [gmr_stack_tmp.data, gmr_stack_tmp.pos, next_actions, next_gmr_mask],
            [grammar_stack.data, grammar_stack.pos, next_inputs.action, next_inputs.gmr_mask])

    layers.utils.map_structure(layers.assign,
            [new_gmr_stack_data, new_gmr_stack_pos, next_actions, new_gmr_mask],
            [grammar_stack.data, grammar_stack.pos, next_inputs.action, next_inputs.gmr_mask])
    layers.logical_or(finished,
            layers.logical_and(condition, layers.logical_not(valid_pos)),
            out=finished)


def _process_type_midd(condition, decoder, grammar_stack, next_inputs, predicted_ids):
    """Process when output type is MID

    Args:
        condition (TYPE): NULL
        decoder (TYPE): NULL
        grammar_stack (TYPE): NULL
        next_inputs (TYPE): NULL
        predicted_ids (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    midd_pred_ids = fluider.elementwise_mul(predicted_ids, condition, axis=0, force=True)
    ## get grammar desc
    # 解码结果(语法ID)对应的具体语法规则序列。比如解码结果为 SingleSQL，则对应的语法序列为 Select Filter
    # shape = [batch_size, grammar.max_desc_len]
    gmr_desc = decoder.grammar_desc(midd_pred_ids)
    # 语法规则序列的长度，比如 SingleSQL --> Select Filter, 则长度为2
    # shape = [batch_size, 1]
    gmr_desc_lens = decoder.grammar_desc_lens(midd_pred_ids)
    # shape = [batch_size, 1]
    gmr_desc_pos = tensor.zeros_like(gmr_desc_lens)

    ## generate next grammar mask by first token in desc
    next_output = nn_utils.batch_gather(gmr_desc, gmr_desc_pos)
    next_actions = decoder.grammar_action(next_output)
    next_gmr_mask = decoder.grammar_mask(next_output)

    ## push left grammar tokens to stack
    gmr_stack_tmp, gmr_stack_pos_tmp = _push_to_stack(gmr_desc, gmr_desc_pos, gmr_desc_lens, grammar_stack)

    ## save result, while condition is True
    new_gmr_stack, new_gmr_stack_pos, new_actions, new_gmr_mask = nn_utils.ifelse(condition,
            [gmr_stack_tmp, gmr_stack_pos_tmp, next_actions, next_gmr_mask],
            [grammar_stack.data, grammar_stack.pos, next_inputs.action, next_inputs.gmr_mask])
    layers.utils.map_structure(layers.assign,
            [new_gmr_stack, new_gmr_stack_pos, new_actions, new_gmr_mask],
            [grammar_stack.data, grammar_stack.pos, next_inputs.action, next_inputs.gmr_mask])


def _save_predict_output(outputs_array, predicted_ids, finished):
    """save predicted_ids to outputs_array, while finished is false.

    Args:
        outputs_array (TYPE): NULL
        predicted_ids (TYPE): NULL
        finished (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    out_arr_tmp = data_structure.Array.push(outputs_array, predicted_ids, in_place=False)
    new_arr_data, new_arr_pos = nn_utils.ifelse(finished,
            [outputs_array.data, outputs_array.pos],
            [out_arr_tmp.data, out_arr_tmp.pos])
    layers.utils.map_structure(layers.assign,
            [new_arr_data, new_arr_pos],
            [outputs_array.data, outputs_array.pos])


def _check_finished(decoder, next_inputs, finished, outputs_array):
    """check finished instance by next_inputs.action, and
    update finished tag and write END to outputs

    Args:
        decoder (TYPE): NULL
        next_inputs (TYPE): NULL
        finished (TYPE): NULL
        outputs_array (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    act_stop = tensor.fill_constant_batch_size_like(next_inputs.action,
            shape=next_inputs.action.shape, value=decoder._grammar.ACTION_STOP, dtype='int64')
    new_finished = layers.logical_and(layers.equal(next_inputs.action, act_stop),
                                      layers.logical_not(finished))

    end_token_id = tensor.fill_constant_batch_size_like(outputs_array.data,
            shape=[-1], value=decoder._grammar.END, dtype=outputs_array.data.dtype)
    out_data_tmp, out_pos_tmp = data_structure.Array.push(outputs_array, end_token_id, in_place=False)
    new_data, new_pos = nn_utils.ifelse(new_finished,
            [out_data_tmp, out_pos_tmp],
            [outputs_array.data, outputs_array.pos])

    layers.assign(new_data, outputs_array.data)
    layers.assign(new_pos, outputs_array.pos)
    layers.logical_or(finished, new_finished, out=finished)


def decode_with_grammar(decoder, inits, decode_vocab, max_step_num, **kwargs):
    """A modification of paddle.fluid.layers.dynamic_decode(...).
    Dynamic decoding performs :code:`decoder.step()` repeatedly until the returned
    Tensor indicating finished status contains all True values or the number of
    decoding step reachs to :attr:`max_step_num`.
    :code:`decoder.initialize()` would be called once before the decoding loop.
    If the `decoder` has implemented `finalize` method, :code:`decoder.finalize()`
    would be called once after the decoding loop.

    Args:
        decoder(Decoder): An instance of `Decoder`.
        inits(tuple): Argument passed to `decoder.initialize`. 
        decode_vocab(DecoderDynamicVocab): namedtuple(table table_len column column_len value value_len)
        max_step_num(int): The maximum number of steps.
        **kwargs: Additional keyword arguments. Arguments passed to `decoder.step`. 

    Returns:
        tuple: A tuple( :code:`(final_outputs, final_states)` ) including the final \
            outputs and states, both are Tensor or nested structure of Tensor. \
            `final_outputs` has the same structure and data types as \
            :code:`decoder.output_dtype` , and each Tenser in `final_outputs` \
            is the stacked of all decoding steps' outputs, which might be revised \
            by :code:`decoder.finalize` . `final_states` is the counterpart \
            at last time step of initial states returned by :code:`decoder.initialize` , \
            thus has the same structure with it and has tensors with same shapes \
            and data types.
    """
    step_cnt = tensor.fill_constant(shape=[1], dtype="int64", value=1)
    max_step_num_tensor = tensor.fill_constant(shape=[1], dtype="int64", value=max_step_num - 2)

    # shape = [batch_size, beam_size, ...]
    initial_inputs, initial_states, initial_finished = decoder.initialize(inits, decode_vocab)
    global_inputs, global_states, global_finished = (initial_inputs, initial_states, initial_finished)
    inputs = initial_inputs
    states = initial_states

    # 保存输出结果
    outputs_arr_data = tensor.fill_constant_batch_size_like(inputs.input,
            shape=[-1, decoder.beam_size, max_step_num], dtype=decoder.output_dtype.predicted_ids, value=0)
    outputs_arr_pos = tensor.fill_constant_batch_size_like(inputs.input,
            shape=[-1, decoder.beam_size, 1], dtype='int64', value=0)
    outputs_array = data_structure.ArrayData(decoder.merge_batch_beams(outputs_arr_data), 
                                             decoder.merge_batch_beams(outputs_arr_pos))

    sequence_lengths = tensor.cast(tensor.zeros_like(initial_finished), "int64")

    # 按语法解码的相关约束数据结构
    grammar_stack_dat = tensor.fill_constant_batch_size_like(
            inputs.input, shape=[-1, decoder.beam_size, max_step_num * STACK_EXPAND_TIMES], dtype='int64', value=0)
    grammar_stack_pos = tensor.fill_constant_batch_size_like(
            inputs.input, shape=[-1, decoder.beam_size, 1], dtype='int64', value=0)
    grammar_stack = data_structure.StackData(decoder.merge_batch_beams(grammar_stack_dat),
                                             decoder.merge_batch_beams(grammar_stack_pos))

    ############        循环解码，直到全部为 finish 状态        ############
    #   finish 的判断：通过 global_finished/next_finished && max_step_num 判断
    cond = layers.logical_not((layers.reduce_all(initial_finished)))
    while_op = layers.While(cond)
    with while_op.block():
        # step_outputs --> OutputWrapper
        # next_states  --> StateWrapper
        # next_inputs  --> DecoderInputsWrapper
        step_outputs, next_states, next_inputs = decoder.step(inputs, states, **kwargs)
        predicted_ids = step_outputs.predicted_ids
        _save_predict_output(outputs_array, predicted_ids, next_states.finished)

        pred_gmr_type = decoder.grammar_type(predicted_ids)
        cond_type_leaf = layers.equal(pred_gmr_type, decoder.GMR_TYPE.LEAF)
        cond_type_midd = layers.equal(pred_gmr_type, decoder.GMR_TYPE.MID)

        _process_type_leaf(cond_type_leaf, decoder, grammar_stack, next_inputs, next_states.finished)
        _process_type_midd(cond_type_midd, decoder, grammar_stack, next_inputs, predicted_ids)

        ##next_sequence_lengths = layers.elementwise_add(sequence_lengths,
        ##                        tensor.cast(layers.logical_not(global_finished), sequence_lengths.dtype))

        _check_finished(decoder, next_inputs, next_states.finished, outputs_array)

        layers.utils.map_structure(tensor.assign, next_inputs, global_inputs)
        layers.utils.map_structure(tensor.assign, next_states, global_states)
        tensor.assign(next_states.finished, global_finished)
        ##tensor.assign(next_sequence_lengths, sequence_lengths)

        # 更新循环条件
        layers.increment(x=step_cnt, value=1.0, in_place=True)
        layers.logical_and(layers.logical_not(layers.reduce_all(next_states.finished)),
                           layers.less_equal(step_cnt, max_step_num_tensor),
                           cond)

    final_outputs = outputs_array.data
    final_states = global_states

    final_outputs, final_states = decoder.finalize(final_outputs, global_states, sequence_lengths)

    return final_outputs, final_states


if __name__ == "__main__":
    """run some simple test cases"""
    import functools
    from text2sql.utils.debug import executor
    from text2sql.grammar import Grammar
    from text2sql import models
    from text2sql.models import gmr_models

    np.random.seed(1129)

    batch_size = 3
    beam_size = 1
    hidden_size = 10
    max_step_num = 10
    stack_size = max_step_num * STACK_EXPAND_TIMES

    # 构建 decoder 对象及相关参数
    cell = models.LSTMDecoderCell(hidden_size)
    grammar = Grammar('data/grammar.txt')
    fn_emb = lambda x: fluid.embedding(x, size=[grammar.vocab_size, hidden_size])
    output_layer = functools.partial(gmr_models.grammar_output, name='decoder_output')
    decoder = gmr_models.GrammarInferDecoder(cell=cell, beam_size=beam_size, grammar=grammar,
                                             fn_embedding=fn_emb, fn_output=output_layer)

    init_states = (layers.data(name='init_hidden', shape=[-1, hidden_size], dtype='float32'),
                   layers.data(name='init_state', shape=[-1, hidden_size], dtype='float32'))
    decode_vocab = gmr_models.DecoderDynamicVocab(
                    layers.data(name='table', shape=[-1, grammar.max_table, hidden_size], dtype='float32'),
                    layers.data(name='table_len', shape=[-1], dtype='int64'),
                    layers.data(name='column', shape=[-1, grammar.max_column, hidden_size], dtype='float32'),
                    layers.data(name='column_len', shape=[-1], dtype='int64'),
                    layers.data(name='value', shape=[-1, grammar.max_value, hidden_size], dtype='float32'),
                    layers.data(name='value_len', shape=[-1], dtype='int64'))

    # test full procedure
    outputs, states = decode_with_grammar(decoder, init_states, decode_vocab, max_step_num=max_step_num)

    cond_end = layers.data(name='cond_end', shape=[-1, 1], dtype='bool')
    cond_const = layers.data(name='cond_const', shape=[-1, 1], dtype='bool')
    cond_leaf = layers.data(name='cond_leaf', shape=[-1, 1], dtype='bool')
    cond_midd = layers.data(name='cond_midd', shape=[-1, 1], dtype='bool')
    cond_to_stack = layers.data(name='cond_to_stack', shape=[-1, 1], dtype='bool')

    step_gmr_desc = layers.data(name='step_gmr_desc', shape=[-1, beam_size, grammar.max_desc_len], dtype='int64')
    step_gmr_pos = layers.data(name='step_gmr_pos', shape=[-1, beam_size, 1], dtype='int64')
    step_gmr_lens = layers.data(name='step_gmr_lens', shape=[-1, beam_size, 1], dtype='int64')
    gmr_stack_dat = layers.data(name='gmr_stack_dat', shape=[-1, beam_size, stack_size], dtype='int64')
    gmr_stack_pos = layers.data(name='gmr_stack_pos', shape=[-1, beam_size, 1], dtype='int64')
    next_finished = layers.data(name='next_finished', shape=[-1, beam_size, 1], dtype='bool')
    outputs_array = layers.data(name='outputs_array', shape=[-1, beam_size, max_step_num], dtype='int64')
    outputs_pos = layers.data(name='outputs_pos', shape=[-1, beam_size, 1], dtype='int64')
    output_tag = layers.data(name='output_tag', shape=[-1, beam_size, 1], dtype='bool')
    gmr_mask = layers.data(name='gmr_mask', shape=[-1, beam_size, 61], dtype='float32')
    input_params = [(step_gmr_desc, step_gmr_pos, step_gmr_lens),
                    (gmr_stack_dat, gmr_stack_pos),
                    next_finished,
                    (outputs_array, outputs_pos, output_tag, gmr_mask)]

    ### test each branch
    #decoder.grammar_decode_wrapper(_process_type_leaf, decoder, cond_leaf, *input_params)
    #decoder.grammar_decode_wrapper(_process_type_midd, decoder, cond_midd, *input_params)

    _data = {
            "init_hidden": np.random.normal(size=[batch_size, hidden_size]).astype(np.float32),
            "init_state":  np.random.normal(size=[batch_size, hidden_size]).astype(np.float32),
            "table":       np.random.normal(size=[batch_size, grammar.max_table, hidden_size]).astype(np.float32),
            "table_len":   np.array([9, 2, 3]).astype(np.int64),
            "column":      np.random.normal(size=[batch_size, grammar.max_column, hidden_size]).astype(np.float32),
            "column_len":  np.array([8, 4, 3]).astype(np.int64),
            "value":       np.random.normal(size=[batch_size, grammar.max_value, hidden_size]).astype(np.float32),
            "value_len":   np.array([5, 7, 3]).astype(np.int64),
            "cond_end":    np.random.randint(0, 2, size=[batch_size, beam_size, 1], dtype='bool'),
            "cond_const":  np.random.randint(0, 2, size=[batch_size, beam_size, 1], dtype='bool'),
            "cond_leaf":   np.random.randint(0, 2, size=[batch_size, beam_size, 1], dtype='bool'),
            "cond_midd":   np.random.randint(0, 2, size=[batch_size, beam_size, 1], dtype='bool'),
            "cond_to_stack": np.random.randint(0, 2, size=[batch_size, beam_size, 1], dtype='bool'),
            "step_gmr_desc": np.random.randint(0, 20, size=[batch_size, beam_size, grammar.max_desc_len], dtype='int64'),
            "step_gmr_pos":  np.random.randint(0, 5, size=[batch_size, beam_size, 1], dtype='int64'),
            "step_gmr_lens": np.random.randint(0, 5, size=[batch_size, beam_size, 1], dtype='int64'),
            "gmr_stack_dat": np.random.randint(0, 20, size=[batch_size, beam_size, stack_size], dtype='int64'),
            "gmr_stack_pos": np.random.randint(0, 5, size=[batch_size, beam_size, 1], dtype='int64'),
            "next_finished": np.random.randint(0, 2, size=[batch_size, beam_size, 1], dtype='bool'),
            "outputs_array": np.zeros(shape=[batch_size, beam_size, max_step_num], dtype='int64'),
            "outputs_pos":   np.random.randint(0, 1, size=[batch_size, beam_size, 1], dtype='int64'),
            "output_tag":    np.random.randint(0, 2, size=[batch_size, beam_size, 1], dtype='bool'),
            "gmr_mask":      np.random.normal(0, 2, size=[batch_size, beam_size, 61]),
        }

    exe = executor.Executor()
    result = exe.run(feed=_data, fetch_list=[outputs_array])
    for var in result:
        print(var)

    #fluid.io.save_inference_model(dirname='output/debug/save_infer',
    #                              feeded_var_names=['table'],
    #                              target_vars=[outputs],
    #                              executor=exe.exe)

