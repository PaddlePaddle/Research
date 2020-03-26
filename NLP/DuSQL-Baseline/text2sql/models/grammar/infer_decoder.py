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

"""grammar based decoder in seq2seq model
"""

import sys
import os
import traceback
import logging
from collections import namedtuple

import numpy as np
from paddle import fluid
from paddle.fluid import layers

from text2sql.utils import fluider
from text2sql.utils import nn_utils
from text2sql.grammar import Grammar
from text2sql.models.grammar import DecoderInputsWrapper
from text2sql.models.grammar import DecoderDynamicVocab

INF = 1e9

# The structure for the returned value outputs of decoder.step.
OutputWrapper = namedtuple("OutputWrapper", "scores predicted_ids parent_ids")
# The structure for the argument states of decoder.step.
StateWrapper = namedtuple("StateWrapper", "cell_states log_probs finished lengths valid_table_mask")

class GrammarInferDecoder(layers.Decoder):

    """Grammar based Decoder in Seq2Seq Model"""

    def __init__(self, cell, beam_size, grammar, fn_output, fn_embedding):
        """init of class

        Args:
            cell (RNNCell/LSTMCell/GRUCell): NULL
            beam_size (int): shoud greater than 0. otherwise, it'll be setted to 1.
                             Currently, only 1 is supported.
            grammar (Grammar): NULL
            fn_output (callable): optional.
            fn_embedding (callable): optional.

        """
        super(GrammarInferDecoder, self).__init__()

        self._cell = cell
        self._beam_size = beam_size
        self._start_token = grammar.START
        self._end_token = grammar.END
        self._init_target_token = grammar.SQL
        self._grammar = grammar
        self._fn_output = fn_output
        self._fn_embedding = lambda x: layers.squeeze(fn_embedding(layers.unsqueeze(x, [1])), [1])

        if self._beam_size != 1:
            raise ValueError('Currently, only beam_size = 1 is supported. got %d' % (self._beam_size))

        # They'll be initialized in self.initialize(...)
        self._batch_size = None
        self._start_token_tensor = None
        self._end_token_tensor = None
        # They'll be initialized in self._grammar_step(...)
        self._vocab_size = None
        self._vocab_size_tensor = None
        self._noend_mask_tensor = None

        if self._beam_size <= 0:
            logging.warn("beam size shoud be greater than 0, but got %d.", self._beam_size)
            self._beam_size = 1

    def initialize(self, init_cell_states, decode_vocab):
        """Generates the input and state for the first decoding step, and the
        initial finished status of each sequence in batch. It shold be called 
        once before the decoding iterations.

        Args:
            init_cell_states ([Variable, Variable]): [init_hidden, init_state]
            decode_vocab(DecoderDynamicVocab): namedtuple(table table_len column column_len value value_len)

        Returns: TODO

        Raises: NULL
        """
        state_tmp = init_cell_states[0]
        # Its type is paddle Variable, not python int.
        self._batch_size = layers.shape(state_tmp)[0]

        ## 语法相关 Tensor 初始化 ##
        # paddle doesn't support assign np.int64
        trans_dtype = lambda x: layers.cast(layers.assign(x.astype(np.int32)), dtype='int64')
        self._grammar_desc = trans_dtype(self._grammar.gmr_desc_arr)
        self._grammar_desc_len = trans_dtype(self._grammar.gmr_desc_lens)
        self._grammar_type = trans_dtype(self._grammar.gmr2type_arr)
        self._grammar_name = trans_dtype(self._grammar.gmr_token2name_arr)
        self._grammar_action = trans_dtype(self._grammar.gmr_token2action_arr)
        self._grammar_mask = layers.assign(self._grammar.grammar_mask_matrix)
        vocab_expand = layers.utils.map_structure(self._expand_to_beam_size, decode_vocab)
        vocab_merge_bb = layers.utils.map_structure(self.merge_batch_beams, vocab_expand)
        self._decode_vocab = DecoderDynamicVocab(*list(vocab_merge_bb))

        # 4 种 grammar 元素类型，用于解码过程中分支的判断
        GrammarType = namedtuple('GrammarType', 'LEAF MID END CONST')
        def _create_gmr_type_tensor(self, type_id):
            """create grammar type tensor with shape=[batch_size * beam_size]

            Args:
                type_id (TYPE): NULL

            Returns: Variable
                     shape = [batch_size * beam_size]
                     dtype = int64

            Raises: NULL
            """
            shape = [self._batch_size, self._beam_size]
            output = layers.fill_constant(shape=shape, value=type_id, dtype='int64')
            return self.merge_batch_beams(output)
        self.GMR_TYPE = GrammarType(_create_gmr_type_tensor(self, self._grammar.TYPE_LEAF),
                                    _create_gmr_type_tensor(self, self._grammar.TYPE_MID),
                                    _create_gmr_type_tensor(self, self._grammar.TYPE_END),
                                    _create_gmr_type_tensor(self, self._grammar.TYPE_CONST))

        self._start_token_tensor = layers.fill_constant(shape=[1], dtype="int64", value=self._start_token)
        self._end_token_tensor = layers.fill_constant(shape=[1], dtype="int64", value=self._end_token)

        ########    init inputs             ########
        # 将 self.start_token_tensor 扩展为第一个时间步输入
        # shape = [batch_size * beam_size]
        init_input_ids = layers.expand(self._start_token_tensor, [self._batch_size * self._beam_size])
        # shape = [batch_size, beam_size, hidden_size]
        init_inputs = self._fn_embedding(init_input_ids)
        init_actions = self.grammar_action(init_input_ids, has_beam=False)

        init_target_token_tensor = layers.fill_constant(shape=[1], dtype="int64", value=self._init_target_token)
        init_target_ids = layers.expand(init_target_token_tensor, [self._batch_size * self._beam_size])
        init_gmr_mask = self.grammar_mask(init_target_ids, has_beam=False)

        ########    init decoding states    ########
        # shape = [batch_size, beam_size, hidden_size]
        init_states = layers.utils.map_structure(self._expand_to_beam_size, init_cell_states)
        init_states = layers.utils.map_structure(self.merge_batch_beams, init_states)
        # shape = [batch_size, beam_size]
        log_probs = layers.expand(layers.assign(np.array([[0.] + [-INF] * (self._beam_size - 1)], dtype="float32")),
                              [self._batch_size, 1])
        log_probs = self.merge_batch_beams(log_probs)
        # global finished tag
        # shape = [batch_size, beam_size]
        init_finished = layers.fill_constant_batch_size_like(log_probs, shape=[-1], dtype="bool", value=False)
        # shape = [batch_size, beam_size]
        init_lengths = layers.zeros_like(init_input_ids)

        valid_table_mask = layers.zeros(
                shape=[self._batch_size * self._beam_size, self._grammar.MAX_TABLE], dtype='float32')
        return (DecoderInputsWrapper(init_inputs, init_actions, init_gmr_mask),
                StateWrapper(init_states, log_probs, init_finished, init_lengths, valid_table_mask),
                init_finished)

    def step(self, inputs, states, **kwargs):
        """Perform a decode step, which can limit decoding vocab space according
        to grammars.

        Args:
            inputs (DecoderInputsWrapper): NULL
            states (TYPE): NULL
            **kwargs (TYPE): NULL

        Returns: (OutputWrapper, StateWrapper, DecoderInputsWrapper)
                 meaning (decode_outputs, decode_states, next_inputs)

        Raises: NULL

        """
        step_inputs, actions, gmr_mask = inputs

        # cell_outputs: [batch_size * beam_size, vocab_size]
        # next_cell_states: []
        cell_outputs, next_cell_states = self._cell(step_inputs, states.cell_states, **kwargs)

        decode_outputs, decode_states = self._grammar_step(cell_outputs, next_cell_states, states, actions, gmr_mask)
        next_input_var = self._fn_embedding(decode_outputs.predicted_ids)
        ## Thel'll be updated while processing grammar decoding.
        next_actions = actions
        next_gmr_mask = gmr_mask
        next_inputs = DecoderInputsWrapper(next_input_var, next_actions, next_gmr_mask)

        return decode_outputs, decode_states, next_inputs

    def finalize(self, outputs, final_states, sequence_lengths):
        """Finialize all decoding process. It should be called once after
        the decoding iterations.

        Args:
            outputs (OutputWrapper): NULL
            final_states (TYPE): NULL
            sequence_lengths (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        ## Currently, only beam_size = 1 is support, so simply delete beam dim here.
        outputs = layers.reshape(outputs, shape=[-1, outputs.shape[-1]])
        return outputs, final_states
        
    @property
    def output_dtype(self):
        """output data type of each item of decoder.step()

        Returns: OutputWrapper, which is a namedtuple
        """
        return OutputWrapper(scores="float32", predicted_ids="int64", parent_ids="int64")

    def grammar_decode_wrapper(self, fn, decoder, condition, step_gmr_info, gmr_stack_info, finished, outputs_info):
        """wrapper of grammar decoding, while process each grammar branch

        Args:
            fn (TYPE): NULL
            decoder (TYPE): NULL
            condition (TYPE): NULL
            step_gmr_info (TYPE): NULL
            gmr_stack_info (TYPE): NULL
            finished (TYPE): NULL
            outputs_info (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        condition = layers.utils.map_structure(self.merge_batch_beams, condition)

        ## 用 new_xx 保存中间结果，以便处理结束后 assign 回原变量
        new_step_gmr_info = layers.utils.map_structure(self.merge_batch_beams, step_gmr_info)
        new_gmr_stack_info = layers.utils.map_structure(self.merge_batch_beams, gmr_stack_info)
        new_finished = layers.utils.map_structure(self.merge_batch_beams, finished)
        new_outputs_info = layers.utils.map_structure(self.merge_batch_beams, outputs_info)

        new_step_gmr_info, new_gmr_stack_info, new_finished, new_outputs_info = \
                fn(decoder, condition, new_step_gmr_info, new_gmr_stack_info, new_finished, new_outputs_info)

        new_step_gmr_info = layers.utils.map_structure(self.split_batch_beams, new_step_gmr_info)
        new_gmr_stack_info = layers.utils.map_structure(self.split_batch_beams, new_gmr_stack_info)
        new_finished = layers.utils.map_structure(self.split_batch_beams, new_finished)
        new_outputs_info = layers.utils.map_structure(self.split_batch_beams, new_outputs_info)

        # 计算结果 assign 回原变量
        layers.utils.map_structure(lambda x, y: layers.assign(x, y), new_step_gmr_info, step_gmr_info)
        layers.utils.map_structure(lambda x, y: layers.assign(x, y), new_gmr_stack_info, gmr_stack_info)
        layers.utils.map_structure(lambda x, y: layers.assign(x, y), new_finished, finished)
        layers.utils.map_structure(lambda x, y: layers.assign(x, y), new_outputs_info, outputs_info)
        return step_gmr_info, gmr_stack_info, finished, outputs_info

    def grammar_desc(self, predicted):
        """get grammar sequence grammar id

        Args:
            predicted (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        #predicted = self.merge_batch_beams(predicted)
        output = layers.gather(self._grammar_desc, predicted)
        #output = self.split_batch_beams(output)
        return output

    def grammar_desc_lens(self, predicted, has_beam=False):
        """get grammar sequence lengths by grammar id

        Args:
            predicted (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        if has_beam:
            predicted = self.merge_batch_beams(predicted)
        output = layers.gather(self._grammar_desc_len, predicted)
        if has_beam:
            output = self.split_batch_beams(output)
        return output

    def grammar_type(self, predicted, has_beam=False):
        """get grammar type by grammar id

        Args:
            predicted (TYPE): NULL
            has_beam (bool): merge batch-beam dimensions. Default is False.

        Returns: TODO

        Raises: NULL

        """
        if has_beam:
            predicted = self.merge_batch_beams(predicted)
        output = layers.gather(self._grammar_type, predicted)
        if has_beam:
            output = self.split_batch_beams(output)
        return output

    def grammar_action(self, predicted, has_beam=False):
        """get grammar type by grammar id

        Args:
            predicted (TYPE): NULL
            has_beam (bool): merge batch-beam dimensions. Default is False.

        Returns: TODO

        Raises: NULL

        """
        if has_beam:
            predicted = self.merge_batch_beams(predicted)
        output = layers.gather(self._grammar_action, predicted)
        if has_beam:
            output = self.split_batch_beams(output)
        return output

    def grammar_mask(self, grammar_id, has_beam=False):
        """get grammar type by grammar id

        Args:
            grammar_id (TYPE): NULL
            has_beam (bool): input/output has beam dim or not.
                             if True, merge batch-beam dimensions of input and split batch-beam dimensions of output.
                             Default is False.

        Returns: TODO

        Raises: NULL

        """
        if has_beam:
            grammar_id = self.merge_batch_beams(grammar_id)
        name_id = layers.gather(self._grammar_name, grammar_id)
        output = layers.gather(self._grammar_mask, name_id)
        if has_beam:
            output = self.split_batch_beams(output)

        return output

    def step_gmr_type(self, gmr_seq, gmr_pos):
        """get type of grammar on gmr_pos of gmr_seq

        Args:
            gmr_seq (TYPE): NULL
            gmr_pos (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        gmr_id = nn_utils.batch_gather(gmr_seq, gmr_pos)
        output = self.grammar_type(gmr_id, False)
        return output

    def _grammar_step(self, logits, next_cell_states, decode_states, actions, gmr_mask):
        """跟进文法约束完成一步解码逻辑

        Args:
            logits (Variable): shape = [batch_size, beam_size, vocab_size]
            next_cell_states (Variable): NULL
            decode_states (StateWrapper): NULL

        Returns: TODO

        Raises: NULL

        """
        # 解码出符合语法规则的 token logits
        logits, valid_table_mask = self._output_layer(logits, actions, gmr_mask, decode_states.valid_table_mask)

        # 初始化 vocab size
        self._vocab_size = logits.shape[-1]
        self._vocab_size_tensor = layers.fill_constant(shape=[1], dtype='int64', value=logits.shape[-1])

        # 计算 log probs，并 mask 掉 finished 部分
        step_log_probs = layers.log(layers.softmax(logits))
        step_log_probs = self._mask_finished_probs(step_log_probs, decode_states.finished)

        scores = layers.reshape(step_log_probs, [-1, self._beam_size * self._vocab_size])
        topk_scores, topk_indices = layers.topk(input=scores, k=self._beam_size)
        topk_scores = layers.reshape(topk_scores, shape=[-1])
        topk_indices = layers.reshape(topk_indices, shape=[-1])

        # top-k 对应的 beam
        beam_indices = layers.elementwise_floordiv(topk_indices, self._vocab_size_tensor)
        # top-k 对应的 token id
        token_indices = layers.elementwise_mod(topk_indices, self._vocab_size_tensor)

        # 根据 top k 的来源，重新组织 step_log_probs
        next_log_probs = nn_utils.batch_gather(
                layers.reshape(step_log_probs, [-1, self._beam_size * self._vocab_size]),
                topk_indices)
        def _beam_gather(x, beam_indices):
            """reshape x to beam dim, and gather each beam_indices
            Args:
                x (TYPE): NULL
            Returns: Variable
            """
            x = self.split_batch_beams(x)
            return nn_utils.batch_gather(x, beam_indices)
        next_cell_states = layers.utils.map_structure(lambda x: _beam_gather(x, beam_indices),
                                                      next_cell_states)
        next_finished = _beam_gather(decode_states.finished, beam_indices)
        next_lens = _beam_gather(decode_states.lengths, beam_indices)

        next_lens = layers.elementwise_add(next_lens,
                layers.cast(layers.logical_not(next_finished), next_lens.dtype))
        next_finished = layers.logical_or(next_finished,
                layers.equal(token_indices, self._end_token_tensor))

        decode_output = OutputWrapper(topk_scores, token_indices, beam_indices)
        decode_states = StateWrapper(next_cell_states, next_log_probs, next_finished, next_lens, valid_table_mask)

        return decode_output, decode_states

    def _output_layer(self, inputs, actions, gmr_mask, valid_table_mask):
        """wrapper of self._fn_output: 增加seq_len 维度，以便 fn_output 正常计算

        Args:
            inputs (Variable): shape = [batch_size * beam_size, hidden_size]
            actions (Variable): shape = [batch_size * beam_size, 1]
            gmr_mask (Variable): shape = [batch_size * beam_size, grammar_size]
            valid_table_mask (Variable): shape = [batch_size * beam_size, max_table]

        Returns: TODO

        Raises: NULL
        """
        # 添加 seq_len 维
        inputs = layers.unsqueeze(inputs, [1])
        actions = layers.unsqueeze(actions, [1])
        gmr_mask = layers.unsqueeze(gmr_mask, [1])

        output, valid_table_mask = self._fn_output(
                inputs, actions, gmr_mask, valid_table_mask, self._decode_vocab, self._grammar)
        return layers.squeeze(output, [1]), valid_table_mask

    def split_batch_beams(self, var):
        """split the first dimension to batch and beam

        Args:
            var (Variable): with shape [batch_size * beam_size, ...]

        Returns: Variable
            with shape [batch_size, beam_size, ...], whoes data and data type is same as var

        Raises: NULL

        """
        return nn_utils.split_first_dim(var, self._beam_size)

    def merge_batch_beams(self, var):
        """merge batch dimension and beam dimension

        Args:
            var (Variable): with shape [batch_size, beam_size, ...]

        Returns: Variable
            with shape [batch_size * beam_size, ...], whoes data and data type is same as var

        Raises: NULL

        """
        return nn_utils.merge_first_ndim(var, n=2)

    def _expand_to_beam_size(self, var):
        """copy form fluid.layers.rnn.py
        This function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed
        of minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a
        shape `[batch_size, beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Args:
            var (Variable): with shape [batch_size, ...]

        Returns: Variable with shape [batch_size, beam_size, ...]

        Raises: NULL

        """
        var = layers.unsqueeze(var, [1])
        expand_times = [1] * len(var.shape)
        expand_times[1] = self._beam_size
        var = layers.expand(var, expand_times)
        return var

    def _mask_finished_probs(self, probs, finished):
        """mask finished beams. it makes
            1. all finished beams probs to be -inf, except end_token which is 0
            2. unfinished beams to remain unchanged

        Args:
            probs (Variable): with shape [batch_size, vocab_size]
            finished (Variable): with shape [batch_size]

        Returns: Variable

        Raises: NULL

        """
        # 初始化 no-end mask
        noend_array = [-INF] * self._vocab_size
        noend_array[self._end_token] = 0
        self._noend_mask_tensor = layers.assign(np.array(noend_array, "float32"))

        finished = layers.cast(finished, dtype=probs.dtype)
        # finished --> 0; not finished --> -1
        not_finished = fluider.increment(finished, value=-1)
        # shape = [batch_size, vocab_size]
        finished_expended = layers.expand(layers.unsqueeze(finished, [1]), [1, self._vocab_size])
        probs = layers.elementwise_mul(finished_expended, self._noend_mask_tensor, axis=-1) - \
                layers.elementwise_mul(probs, not_finished, axis=0)
        return probs

    @property
    def beam_size(self):
        """read property of beam_size"""
        return self._beam_size
    

if __name__ == "__main__":
    """run some simple test cases"""
    import functools
    from text2sql.utils.debug import executor
    from text2sql.grammar import Grammar
    from text2sql import models
    from text2sql.models import gmr_models

    B = 3           # batch_size
    H = 10          # hidden_size
    beam_size = 1

    # 构建 decoder 对象及相关参数
    cell = models.RNNDecodeCell(H)
    grammar = Grammar('conf/grammar.txt')
    fn_emb = lambda x: fluid.embedding(x, size=[grammar.vocab_size, H])
    output_layer = functools.partial(gmr_models.grammar_output, name='decoder_output')
    decoder = GrammarInferDecoder(cell=cell, beam_size=beam_size, grammar=grammar,
                             fn_output=output_layer, fn_embedding=fn_emb)

    init_states = (layers.data(name='init_hidden', shape=[-1, H], dtype='float32'),
                   layers.data(name='init_state', shape=[-1, H], dtype='float32'))
    #actions = layers.data(name='actions', shape=[-1, beam_size], dtype='int64')
    #gmr_mask = layers.data(name='gmr_mask', shape=[-1, beam_size, grammar.grammar_size], dtype='float32')
    decode_vocab = DecoderDynamicVocab(
            layers.data(name='table', shape=[-1, grammar.MAX_TABLE, H], dtype='float32'),
            layers.data(name='table_len', shape=[-1], dtype='int64'),
            layers.data(name='column', shape=[-1, grammar.MAX_COLUMN, H], dtype='float32'),
            layers.data(name='column_len', shape=[-1], dtype='int64'),
            layers.data(name='value', shape=[-1, grammar.MAX_VALUE, H], dtype='float32'),
            layers.data(name='value_len', shape=[-1], dtype='int64'))

    # 初始化，实际代码中只被调用一次
    inputs, states, finished = decoder.initialize(init_states, decode_vocab)
    # 执行一步解码，实际代码中会在 WhileOP 中被循环调用
    outputs, states, next_inputs = decoder.step(inputs=inputs, states=states)

    _data = {
            "init_hidden": np.random.normal(size=[B, H]).astype(np.float32),
            "init_state": np.random.normal(size=[B, H]).astype(np.float32),
            #"actions": np.array([[1, 0], [2, 4], [3, 0]]).astype(np.int64),
            #"gmr_mask": np.random.normal(size=[B, beam_size, grammar.grammar_size]).astype(np.float32),
            "table": np.random.normal(size=[B, grammar.max_table, H]).astype(np.float32),
            "table_len": np.array([9, 3, 2]).astype(np.int64),
            "column": np.random.normal(size=[B, grammar.max_column, H]).astype(np.float32),
            "column_len": np.array([2, 3, 7]).astype(np.int64),
            "value": np.random.normal(size=[B, grammar.max_value, H]).astype(np.float32),
            "value_len": np.array([6, 3, 8]).astype(np.int64),
        }

    exe = executor.Executor()
    result = exe.run(feed=_data, fetch_list=outputs + inputs)
    for var in result:
        print(var)

