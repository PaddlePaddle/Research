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

"""field reader for grammar based rule sequence
"""

import sys
import os
import traceback
import logging

import numpy as np
from text2sql.framework.register import RegisterSet
from text2sql.framework.rule import InstanceName as C
from text2sql.framework.rule import DataShape, FieldLength
from text2sql.framework.reader.field_reader.base_field_reader import BaseFieldReader
from text2sql.framework.reader.util_helper import generate_pad_batch_data

from text2sql.datalib import DName
from text2sql.grammar import Grammar

GRAMMAR_LABEL_FIELD_NUM = 7
INF = 1e9

@RegisterSet.field_reader.register
class GrammarLabelFieldReader(BaseFieldReader):
    """seq2seq label的专用field_reader
    """
    def __init__(self, field_config):
        """
        Args:
            field_config (TYPE):
        """
        BaseFieldReader.__init__(self, field_config=field_config)
        self.paddle_version_code = 1.6

        if self.field_config.tokenizer_info:
            tokenizer_class = RegisterSet.tokenizer.__getitem__(self.field_config.tokenizer_info["type"])
            params = None
            if self.field_config.tokenizer_info.__contains__("params"):
                params = self.field_config.tokenizer_info["params"]
            self.tokenizer = tokenizer_class(vocab_file=self.field_config.vocab_path,
                                             split_char=self.field_config.tokenizer_info["split_char"],
                                             unk_token=self.field_config.tokenizer_info["unk_token"],
                                             params=params)
            self.grammar = Grammar(self.field_config.tokenizer_info['grammar_file'])

    def init_reader(self):
        """ 初始化reader格式

        Returns: tuple
            reader的shape[]、type[]、level[]
        """
        shape = []
        types = []
        levels = []
        # src_ids
        shape.append([-1, self.field_config.max_seq_len])
        levels.append(0)
        types.append('int64')
        # infer_tar_ids
        shape.append([-1, self.field_config.max_seq_len])
        levels.append(0)
        types.append('int64')
        # mask_ids
        shape.append([-1, self.field_config.max_seq_len])
        levels.append(0)
        types.append('float32')
        # seq_lens
        shape.append([-1])
        levels.append(0)
        types.append('int64')
        # infer vocab mask
        shape.append([-1, self.field_config.max_seq_len])
        levels.append(0)
        types.append('float32')
        # infer grammar mask
        shape.append([-1, self.field_config.max_seq_len, self.grammar.grammar_size])
        levels.append(0)
        types.append('float32')
        # infer col2table mask
        shape.append([-1, self.field_config.max_seq_len, self.grammar.MAX_TABLE])
        levels.append(0)
        types.append('float32')

        return shape, types, levels

    def convert_texts_to_ids(self, batch_text):
        """处理batch 内的样本，转换为模型训练需要的格式和数据

        Args:
            batch_text (list): [batch_label, batch_table_name, batch_column_name, batch_value]

        Returns: TODO

        """
        train_src_ids = []
        infer_src_ids = []
        train_label_ids = []
        infer_label_ids = []
        label_mask = []
        label_lens = []
        table_cnt = []
        column_cnt = []
        value_cnt = []
        infer_col2table_mask = []
        for label, table, column, value, col2table in zip(*batch_text):
            col2table = col2table.tolist()
            # 预处理阶段直接转为ID作为输入
            src_id = [int(x) for x in label.split(" ")]

            # 加上截断策略
            if len(src_id) > self.field_config.max_seq_len - 1:
                logging.warn('input is too long: %d > %d', len(src_id), self.field_config.max_seq_len - 1)
                src_id = src_id[0:self.field_config.max_seq_len - 1]

            train_src_id = [self.grammar.START] + src_id
            infer_src_id = src_id + [self.grammar.END]
            pad_len = self.field_config.max_seq_len - len(train_src_id)
            pad_ids = [0] * pad_len

            train_src_ids.append(train_src_id)
            infer_src_ids.append(infer_src_id)
            train_label_ids.append(train_src_id + pad_ids)
            infer_label_ids.append(infer_src_id + pad_ids)
            label_mask.append([1] * len(train_src_id) + pad_ids)
            label_lens.append(len(train_src_id))

            table_cnt.append(table.count('[SEP]') + 1)
            column_cnt.append(column.count('[SEP]') + 1)
            value_cnt.append(value.count('[SEP]') + 1)
            infer_col2table_mask.append(self._gen_col2table_mask(infer_src_id, col2table, pad_len))

        train_label_ids = np.array(train_label_ids).astype(np.int64).reshape([-1, self.field_config.max_seq_len])
        infer_label_ids = np.array(infer_label_ids).astype(np.int64).reshape([-1, self.field_config.max_seq_len])
        label_mask = np.array(label_mask).astype(np.float32)
        label_lens = np.array(label_lens).astype(np.int64)
        infer_col2table_mask = np.array(infer_col2table_mask).astype(np.float32)

        infer_actions, infer_gmr_mask = self._create_infer_gmr_mask(
                infer_label_ids, label_lens, (table_cnt, column_cnt, value_cnt))

        return_list = []
        return_list.append(train_label_ids)
        return_list.append(infer_label_ids)
        return_list.append(label_mask)
        return_list.append(label_lens)
        return_list.append(infer_actions)
        return_list.append(infer_gmr_mask)
        return_list.append(infer_col2table_mask)

        return return_list

    def structure_fields_dict(self, slots_id, start_index, need_emb=False):
        """静态图调用的方法，生成一个dict。目前仅包含各个字段的ID序列

        Args:
            slots_id (TYPE) pyreader输出的完整的id序列
            start_index (TYPE) 当前需要处理的field在slot_id_list中的起始位置
            need_emb (bool) 是否需要embedding（预测过程中是不需要embedding的）

        Returns: TODO
        """
        record_id_dict = {}
        record_id_dict[DName.TRAIN_LABEL] = slots_id[start_index]
        record_id_dict[DName.INFER_LABEL] = slots_id[start_index + 1]
        record_id_dict[DName.MASK_IDS] = slots_id[start_index + 2]
        record_id_dict[DName.SEQ_LENS] = slots_id[start_index + 3]
        record_id_dict[DName.INFER_ACTIONS] = slots_id[start_index + 4]
        record_id_dict[DName.INFER_GMR_MASK] = slots_id[start_index + 5]
        record_id_dict[DName.INFER_COL2TABLE_MASK] = slots_id[start_index + 6]

        record_dict = {}
        record_dict[C.RECORD_ID] = record_id_dict
        record_dict[C.RECORD_EMB] = None

        return record_dict

    def get_field_length(self):
        """获取当前这个field在进行了序列化之后，在slot_id_list中占多少长度
        :return:
        """
        return GRAMMAR_LABEL_FIELD_NUM

    def _create_infer_gmr_mask(self, batch_seq_ids, batch_seq_lens, db_info_lens):
        """create vocab mask according to grammar

        Args:
            batch_seq_ids (TYPE): NULL
            batch_seq_lens (TYPE): NULL
            db_info_lens (TYPE): (table_lens, column_lens, value_lens)

        Returns: TODO

        Raises: NULL
        """
        max_seq_len = len(batch_seq_ids[0])
        lst_batch_action = []
        lst_batch_mask = []
        for batch_idx, instance in enumerate(batch_seq_ids):
            ins_len = batch_seq_lens[batch_idx]
            lst_ins_action = []
            lst_ins_mask = []
            for token_idx, token_id in enumerate(instance):
                if token_idx >= ins_len:
                    break
                gmr_name = self.grammar.gmr2name_arr[token_id]
                lst_ins_action.append(self.grammar.gmr_name2action[gmr_name])
                if gmr_name in (self.grammar.GMR_NAME_T, self.grammar.GMR_NAME_C, self.grammar.GMR_NAME_V):
                    vocab_mask = [-INF] * self.grammar.grammar_size
                else:    # grammar variables
                    vocab_mask = self.grammar.grammar_mask_matrix[gmr_name].tolist()
                lst_ins_mask.append(vocab_mask)
            lst_ins_action.extend([self.grammar.ACTION_NONE] * (max_seq_len - ins_len))
            lst_ins_mask.extend([[-INF] * self.grammar.grammar_size] * (max_seq_len - ins_len))

            lst_batch_action.append(lst_ins_action)
            lst_batch_mask.append(lst_ins_mask)
        return np.array(lst_batch_action).astype(np.int64), np.array(lst_batch_mask).astype(np.float32)

    def _gen_col2table_mask(self, src_ids, col2table_mask, pad_len):
        """generate infer col2table mask

        Args:
            src_ids (TYPE): NULL
            col2table_mask (TYPE): NULL
            pad_len (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        table_id_start = self.grammar.grammar_size
        table_id_end = self.grammar.grammar_size + self.grammar.MAX_TABLE
        column_id_start = table_id_end

        empty_mask = [0.0] * self.grammar.MAX_TABLE
        lst_infer_col2table_mask = []
        for idx, src_id in enumerate(src_ids):
            if src_id >= table_id_start and src_id < table_id_end:
                col_id = src_ids[idx - 1] - column_id_start
                lst_infer_col2table_mask.append(col2table_mask[col_id])
            else:
                lst_infer_col2table_mask.append(empty_mask)

        return lst_infer_col2table_mask + [empty_mask] * pad_len


if __name__ == "__main__":
    """run some simple test cases"""
    field_config = {"name": "label",
                    "data_type": "string",
                    "reader": {
                        "type": "GenerateLabelFieldReader"
                    },
                    "tokenizer": {
                        "type": "CustomTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]"
                    },
                    "need_convert": True,
                    "vocab_path": "./data/vocab_en.txt",
                    "max_seq_len": 128,
                    "truncation_type": 0,
                    "padding_id": 0,
                    }

    reader = GrammarLabelFieldReader(field_config)
