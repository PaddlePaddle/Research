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

"""text2sql ernie field reader
"""

import sys
import os
import traceback
import logging

import numpy as np
from text2sql.framework.rule import InstanceName as C
from text2sql.framework.rule import DataShape
from text2sql.framework.register import RegisterSet
from text2sql.framework.reader.field_reader.base_field_reader import BaseFieldReader

from text2sql.utils import pad_batch_data
from text2sql.datalib import DName

TEXT2SQL_ERNIE_FIELD_NUM = 24
g_meaning_full_star = '数量。任意。整体'

@RegisterSet.field_reader.register
class Text2SQLErnieFieldReader(BaseFieldReader):

    """Ernie Text Field Reader. """

    def __init__(self, field_config):
        """init of class

        Args:
            field_config (TYPE): NULL

        """
        super(Text2SQLErnieFieldReader, self).__init__(field_config)

        self.paddle_version_code = 1.6

        if self.field_config.tokenizer_info:
            tokenizer_class = RegisterSet.tokenizer.__getitem__(self.field_config.tokenizer_info["type"])
            params = None
            if "params" in self.field_config.tokenizer_info:
                params = self.field_config.tokenizer_info["params"]
            self.tokenizer = tokenizer_class(vocab_file=self.field_config.vocab_path,
                                             split_char=self.field_config.tokenizer_info["split_char"],
                                             unk_token=self.field_config.tokenizer_info["unk_token"],
                                             params=params)
        self.max_q_span, self.max_table_name, self.max_column_name, self.max_value_item = \
                [int(x) for x in self.field_config.tokenizer_info["max_item_len"].split(',')]
        self.max_name_tokens = self.field_config.tokenizer_info["max_name_tokens"]

    def init_reader(self):
        """ 初始化reader格式
                :return: reader的shape[]、type[]、level[]
                """
        shape = []
        types = []
        levels = []
        # 1. src_id
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('int64')
        # 2. sentence_id
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('int64')
        # 3. position_id
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('int64')
        # 4. mask_ids
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('float32')
        # 5. task_id
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('int64')
        # 6. seq_lens
        shape.append([-1])
        levels.append(0)
        types.append('int64')
        # 7. src_id
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('int64')
        # 8. sentence_id
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('int64')
        # 9. position_id
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('int64')
        # 10. mask_ids
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('float32')
        # 11. task_id
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('int64')
        # 12. seq_lens
        shape.append([-1])
        levels.append(0)
        types.append('int64')
        # 13. question pos
        shape.append([-1, self.max_q_span, self.max_name_tokens])
        levels.append(0)
        types.append('int64')
        # 14. table name pos
        shape.append([-1, self.max_table_name, self.max_name_tokens])
        levels.append(0)
        types.append('int64')
        # 15. column name pos
        shape.append([-1, self.max_column_name, self.max_name_tokens])
        levels.append(0)
        types.append('int64')
        # 16. value pos
        shape.append([-1, self.max_value_item, self.max_name_tokens])
        levels.append(0)
        types.append('int64')
        # 17. question lens
        shape.append([-1])
        levels.append(0)
        types.append('int64')
        # 18. table lens
        shape.append([-1])
        levels.append(0)
        types.append('int64')
        # 19. column lens
        shape.append([-1])
        levels.append(0)
        types.append('int64')
        # 20. value lens
        shape.append([-1])
        levels.append(0)
        types.append('int64')
        # 21. question span lens
        shape.append([-1, self.max_q_span])
        levels.append(0)
        types.append('int64')
        # 22. table name tokens lens
        shape.append([-1, self.max_table_name])
        levels.append(0)
        types.append('int64')
        # 23. column name tokens lens
        shape.append([-1, self.max_column_name])
        levels.append(0)
        types.append('int64')
        # 24. value item tokens lens
        shape.append([-1, self.max_value_item])
        levels.append(0)
        types.append('int64')
        return shape, types, levels

    def convert_texts_to_ids(self, batch_text):
        """将一个batch的明文text转成id
        :param batch_text:
        :return:
        """
        max_len = self.field_config.max_seq_len

        lst_qtc_id = []
        lst_qtc_position_id = []
        lst_qtc_task_id = []
        lst_qtc_sentence_id = []
        lst_qv_id = []
        lst_qv_position_id = []
        lst_qv_task_id = []
        lst_qv_sentence_id = []
        lst_q_span_pos = []
        lst_q_span_len = []
        lst_t_pos = []
        lst_c_pos = []
        lst_v_pos = []
        lst_q_len = []
        lst_t_len = []
        lst_c_len = []
        lst_v_len = []
        lst_t_toks_len = []
        lst_c_toks_len = []
        lst_v_toks_len = []

        for question, tnames, cnames, values in zip(*batch_text):
            cnames = g_meaning_full_star + ' ' + cnames.split(' ', 1)[1]
            q_toks = self._segment_2d_field(question)
            t_toks = self._segment_2d_field(tnames)
            c_toks = self._segment_2d_field(cnames)
            v_toks = self._segment_2d_field(values)

            base_idx = 1
            q_span_pos, q_span_len = self._gen_all_pos(base_idx, q_toks, del_sep=True)
            base_idx = len(q_toks) + 2   # +2 是增加 [CLS] 和 [SEP] 后的base idx
            t_pos, t_toks_len = self._gen_all_pos(base_idx, t_toks)
            base_idx += len(t_toks)
            c_pos, c_toks_len = self._gen_all_pos(base_idx, c_toks)
            base_idx = len(c_toks) + 2   # +2 是增加 [CLS] 和 [SEP] 后的base idx
            v_pos, v_toks_len = self._gen_all_pos(base_idx, v_toks)

            tokens_qtc = ['[CLS]'] + q_toks + ['[SEP]'] + t_toks + c_toks
            tokens_qv = ['[CLS]'] + q_toks + ['[SEP]'] + v_toks
            # 加上截断策略
            if len(tokens_qtc) > max_len:
                raise ValueError('input tokens num(%d) > max_len(%d)' % (len(tokens_qtc), max_len))
            if len(tokens_qv) > max_len:
                raise ValueError('input tokens num(%d) > max_len(%d): %s' % (len(tokens_qv), max_len, str(values)))

            qtc_id = self.tokenizer.convert_tokens_to_ids(tokens_qtc)
            qv_id = self.tokenizer.convert_tokens_to_ids(tokens_qv)

            qtc_pos_id = list(range(len(qtc_id)))
            qtc_task_id = [0] * len(qtc_id)
            qtc_sentence_id = [0] * len(qtc_id)
            qv_pos_id = list(range(len(qv_id)))
            qv_task_id = [0] * len(qv_id)
            qv_sentence_id = [0] * len(qv_id)

            lst_qtc_id.append(qtc_id)
            lst_qtc_position_id.append(qtc_pos_id)
            lst_qtc_task_id.append(qtc_task_id)
            lst_qtc_sentence_id.append(qtc_sentence_id)
            lst_qv_id.append(qv_id)
            lst_qv_position_id.append(qv_pos_id)
            lst_qv_task_id.append(qv_task_id)
            lst_qv_sentence_id.append(qv_sentence_id)

            lst_q_span_pos.append(q_span_pos)
            lst_t_pos.append(t_pos)
            lst_c_pos.append(c_pos)
            lst_v_pos.append(v_pos)
            lst_q_len.append(len(q_span_pos))
            lst_t_len.append(len(t_pos))
            lst_c_len.append(len(c_pos))
            lst_v_len.append(len(v_pos))
            lst_q_span_len.append(q_span_len)
            lst_t_toks_len.append(t_toks_len)
            lst_c_toks_len.append(c_toks_len)
            lst_v_toks_len.append(v_toks_len)

        return_list = []
        qtc_padded_ids, qtc_input_mask, qtc_seq_lens_batch = pad_batch_data(lst_qtc_id,
                                                                max_len=max_len,
                                                                pad_idx=self.field_config.padding_id,
                                                                return_input_mask=True,
                                                                return_seq_lens=True,
                                                                paddle_version_code=self.paddle_version_code)
        qv_padded_ids, qv_input_mask, qv_seq_lens_batch = pad_batch_data(lst_qv_id,
                                                                max_len=max_len,
                                                                pad_idx=self.field_config.padding_id,
                                                                return_input_mask=True,
                                                                return_seq_lens=True,
                                                                paddle_version_code=self.paddle_version_code)
        qtc_sent_ids_batch = pad_batch_data(lst_qtc_sentence_id, max_len=max_len, pad_idx=self.field_config.padding_id)
        qtc_pos_ids_batch = pad_batch_data(lst_qtc_position_id, max_len=max_len, pad_idx=self.field_config.padding_id)
        qtc_task_id_batch = pad_batch_data(lst_qtc_task_id, max_len=max_len, pad_idx=self.field_config.padding_id)
        qv_sent_ids_batch = pad_batch_data(lst_qv_sentence_id, max_len=max_len, pad_idx=self.field_config.padding_id)
        qv_pos_ids_batch = pad_batch_data(lst_qv_position_id, max_len=max_len, pad_idx=self.field_config.padding_id)
        qv_task_id_batch = pad_batch_data(lst_qv_task_id, max_len=max_len, pad_idx=self.field_config.padding_id)

        q_span_pos_batch = pad_batch_data(lst_q_span_pos,
                shape=[-1, self.max_q_span, self.max_name_tokens], pad_idx=[0] * self.max_name_tokens)
        t_toks_pos_batch = pad_batch_data(lst_t_pos,
                shape=[-1, self.max_table_name, self.max_name_tokens], pad_idx=[0] * self.max_name_tokens)
        c_toks_pos_batch = pad_batch_data(lst_c_pos,
                shape=[-1, self.max_column_name, self.max_name_tokens], pad_idx=[0] * self.max_name_tokens)
        v_toks_pos_batch = pad_batch_data(lst_v_pos,
                shape=[-1, self.max_value_item, self.max_name_tokens], pad_idx=[0] * self.max_name_tokens)
        q_span_len_batch = pad_batch_data(lst_q_span_len, shape=[-1, self.max_q_span])
        t_toks_len_batch = pad_batch_data(lst_t_toks_len, shape=[-1, self.max_table_name])
        c_toks_len_batch = pad_batch_data(lst_c_toks_len, shape=[-1, self.max_column_name])
        v_toks_len_batch = pad_batch_data(lst_v_toks_len, shape=[-1, self.max_value_item])

        return_list.append(qtc_padded_ids)          #1
        return_list.append(qtc_sent_ids_batch)      #2
        return_list.append(qtc_pos_ids_batch)       #3
        return_list.append(qtc_input_mask)          #4
        return_list.append(qtc_task_id_batch)       #5
        return_list.append(qtc_seq_lens_batch)      #6
        return_list.append(qv_padded_ids)           #7
        return_list.append(qv_sent_ids_batch)       #8
        return_list.append(qv_pos_ids_batch)        #9
        return_list.append(qv_input_mask)           #10
        return_list.append(qv_task_id_batch)        #11
        return_list.append(qv_seq_lens_batch)       #12
        return_list.append(q_span_pos_batch)        #13
        return_list.append(t_toks_pos_batch)        #14
        return_list.append(c_toks_pos_batch)        #15
        return_list.append(v_toks_pos_batch)        #16
        return_list.append(lst_q_len)               #17
        return_list.append(lst_t_len)               #18
        return_list.append(lst_c_len)               #19
        return_list.append(lst_v_len)               #20
        return_list.append(q_span_len_batch)        #21
        return_list.append(t_toks_len_batch)        #22
        return_list.append(c_toks_len_batch)        #23
        return_list.append(v_toks_len_batch)        #24

        return return_list

    def structure_fields_dict(self, fields_id, start_index, need_emb=True):
        """静态图调用的方法，生成一个dict， dict有两个key:id , emb. id对应的是pyreader读出来的各个field产出的id，emb对应的是各个
        field对应的embedding
        :param fields_id: pyreader输出的完整的id序列
        :param start_index:当前需要处理的field在field_id_list中的起始位置
        :param need_emb:是否需要embedding（预测过程中是不需要embedding的）
        :return:
        """
        record_id_dict = {}
        record_id_dict[DName.QTC_IDS] = fields_id[start_index]
        record_id_dict[DName.QTC_SENTENCE_IDS] = fields_id[start_index + 1]
        record_id_dict[DName.QTC_POS_IDS] = fields_id[start_index + 2]
        record_id_dict[DName.QTC_MASK_IDS] = fields_id[start_index + 3]
        record_id_dict[DName.QTC_TASK_IDS] = fields_id[start_index + 4]
        record_id_dict[DName.QTC_SEQ_LENS] = fields_id[start_index + 5]
        record_id_dict[DName.QV_IDS] = fields_id[start_index + 6]
        record_id_dict[DName.QV_SENTENCE_IDS] = fields_id[start_index + 7]
        record_id_dict[DName.QV_POS_IDS] = fields_id[start_index + 8]
        record_id_dict[DName.QV_MASK_IDS] = fields_id[start_index + 9]
        record_id_dict[DName.QV_TASK_IDS] = fields_id[start_index + 10]
        record_id_dict[DName.QV_SEQ_LENS] = fields_id[start_index + 11]
        record_id_dict[DName.Q_POS] = fields_id[start_index + 12]
        record_id_dict[DName.T_POS] = fields_id[start_index + 13]
        record_id_dict[DName.C_POS] = fields_id[start_index + 14]
        record_id_dict[DName.V_POS] = fields_id[start_index + 15]
        record_id_dict[DName.Q_LEN] = fields_id[start_index + 16]
        record_id_dict[DName.T_LEN] = fields_id[start_index + 17]
        record_id_dict[DName.C_LEN] = fields_id[start_index + 18]
        record_id_dict[DName.V_LEN] = fields_id[start_index + 19]
        record_id_dict[DName.Q_SPAN_LEN] = fields_id[start_index + 20]
        record_id_dict[DName.T_TOKS_LEN] = fields_id[start_index + 21]
        record_id_dict[DName.C_TOKS_LEN] = fields_id[start_index + 22]
        record_id_dict[DName.V_TOKS_LEN] = fields_id[start_index + 23]

        record_emb_dict = None
        if need_emb and self.token_embedding:
            record_emb_dict = self.token_embedding.get_token_embedding(record_id_dict)

        record_dict = {}
        record_dict[C.RECORD_ID] = record_id_dict
        record_dict[C.RECORD_EMB] = record_emb_dict

        return record_dict

    def get_field_length(self):
        """获取当前这个field在进行了序列化之后，在field_id_list中占多少长度
        :return:
        """
        return TEXT2SQL_ERNIE_FIELD_NUM

    def _segment_2d_field(self, sentence, sep_str='[SEP]'):
        """segment 2d field, like table names, column names

        Args:
            sentence (TYPE): NULL
            sep_str (TYPE): Default is ' [SEP] '

        Returns: TODO

        Raises: NULL
        """
        lst_result = []
        for item_str in sentence.split(' %s ' % (sep_str)):
            tmp_result = self.tokenizer.tokenize(item_str)
            if len(tmp_result) > self.max_name_tokens - 1:
                logging.debug('name tokens size(%d) > max_name_tokens(%d): %s',
                              len(tmp_result), self.max_name_tokens, '/'.join(tmp_result))
                tmp_result = tmp_result[:self.max_name_tokens - 1]
            lst_result += tmp_result + [sep_str]
        return lst_result

    def _gen_all_pos(self, base_idx, tokens, del_sep=False):
        """generate each tokens pos and name lens

        Args:
            base_idx (int): NULL
            tokens (list): e.g. stu name [SEP] class no [SEP] age [SEP] ...
            del_sep (bool): if True, delete [SEP] token in inputs tokens

        Returns: TODO

        Raises: NULL
        """
        lst_pos_result = []
        lst_name_lens = []
        start = 0
        idx = 0
        while idx < len(tokens):
            if tokens[idx] != '[SEP]':
                idx += 1
                continue
            tmp_pos_list = list(range(start, idx))
            tmp_pos_list = [x + base_idx for x in tmp_pos_list]
            lst_name_lens.append(len(tmp_pos_list))
            lst_pos_result.append(tmp_pos_list + [0] * (self.max_name_tokens - len(tmp_pos_list)))
            if del_sep:
                tokens.pop(idx)
            else:
                idx += 1
            start = idx

        return lst_pos_result, lst_name_lens


if __name__ == "__main__":
    """run some simple test cases"""
    pass


