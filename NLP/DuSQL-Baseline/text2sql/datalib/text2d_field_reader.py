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

"""table schema/table name field reader, which are 2 dimentions sequences
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
from text2sql.framework.utils.util_helper import truncation_words

from text2sql.utils import pad_batch_data
from text2sql.datalib import DName

TEXT_2D_FIELD_NUM = 6

@RegisterSet.field_reader.register
class Text2DFieldReader(BaseFieldReader):

    """2D Text Field Reader. """

    def __init__(self, field_config):
        """init of class

        Args:
            field_config (TYPE): NULL


        """
        super(Text2DFieldReader, self).__init__(field_config)
        self.paddle_version_code = 1.6

        tokenizer_class = RegisterSet.tokenizer.__getitem__(self.field_config.tokenizer_info["type"])
        self.tokenizer = tokenizer_class(vocab_file=self.field_config.vocab_path,
                                         split_char=self.field_config.tokenizer_info["split_char"],
                                         unk_token=self.field_config.tokenizer_info["unk_token"])
        self.max_item_len = self.field_config.tokenizer_info["max_item_len"]

        if self.field_config.embedding_info and self.field_config.embedding_info["use_reader_emb"]:
            self.token_embedding = CustomFluidTokenEmbedding(
                                        emb_dim=self.field_config.embedding_info["emb_dim"],
                                        vocab_size=self.tokenizer.vocabulary.get_vocab_size())
        self.max_name_tokens = self.field_config.tokenizer_info["max_name_tokens"]

        if self.paddle_version_code <= 1.5:
            self.seq_len_shape = [-1, 1]
        else:
            self.seq_len_shape = [-1]

    def init_reader(self):
        """ 初始化reader格式
        :return: reader的shape[]、type[]、level[]
        """
        shape = []
        types = []
        levels = []
        if self.field_config.data_type == DataShape.STRING:
            # src_ids
            #shape.append([-1, self.field_config.max_len_1d, self.field_config.max_len_2d, 1])
            shape.append([-1, self.field_config.max_seq_len, 1])
            levels.append(0)
            types.append('int64')
        else:
            raise TypeError("data_type must be string")

        # mask_ids
        #shape.append([-1, self.field_config.max_len_1d, self.field_config.max_len_2d, 1])
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('float32')

        # seq_length
        shape.append([-1])
        levels.append(0)
        types.append('int64')

        # the name length
        shape.append([-1])
        levels.append(0)
        types.append('int64')

        # the name block pos
        shape.append([-1, self.max_item_len, self.max_name_tokens])
        levels.append(0)
        types.append('int64')

        # the name block len
        shape.append([-1, self.max_item_len])
        levels.append(0)
        types.append('int64')

        return shape, types, levels

    def convert_texts_to_ids(self, batch_text):
        """convert a batch of input text instances to ids

        Args:
            batch_text (list of string): NULL

        Returns: TODO

        Raises: NULL

        """
        max_len = self.field_config.max_seq_len

        src_ids = []
        name_len = []
        name_block_pos = []
        name_block_len = []
        sep_id = self.tokenizer.covert_token_to_id("[SEP]")
        unk_id = self.tokenizer.covert_token_to_id("[UNK]")
        for idx_batch, text in enumerate(batch_text):
            if self.field_config.need_convert:
                tokens = self.tokenizer.tokenize(text)
                src_id = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                src_id = text.split(" ")

            # 加上截断策略
            if len(src_id) > max_len - 1:
                logging.warn('input instance is to long(max %d): %s', max_len - 1, text)
                src_id = truncation_words(src_id, max_len - 1, self.field_config.truncation_type)
            if src_id[-1] != sep_id:
                src_id.append(sep_id)

            if src_id.count(sep_id) > self.max_item_len:
                raise ValueError("too many items. expacted max is %d, but got %d" % (self.max_item_len, src_id.count(sep_id)))
            src_ids.append(src_id)

            idx_begin = 0
            block_pos_tmp = []
            block_len_tmp = []
            for idx_end, tid in enumerate(src_id):
                if tid == sep_id:
                    supp_num = self.max_name_tokens - (idx_end - idx_begin)
                    block_pos_tmp.append(list(range(idx_begin, idx_end)) + [0] * supp_num)
                    block_len_tmp.append(idx_end - idx_begin)
                    idx_begin = idx_end + 1
            assert all([x > 0 for x in block_len_tmp]), 'token len should > 0: %s' % text
            name_len.append(len(block_pos_tmp))
            name_block_pos.append(block_pos_tmp)
            name_block_len.append(block_len_tmp)

        return_list = []
        padding_id = self.field_config.padding_id
        padded_ids, mask_ids, batch_seq_lens = pad_batch_data(
                                                    src_ids,
                                                    max_len=max_len,
                                                    pad_idx=padding_id,
                                                    return_input_mask=True,
                                                    return_seq_lens=True,
                                                    paddle_version_code=self.paddle_version_code)

        name_len = np.array(name_len).astype('int64').reshape(self.seq_len_shape)
        batch_name_pos = pad_batch_data(name_block_pos,
                shape=[-1, self.max_item_len, self.max_name_tokens], pad_idx=[0] * self.max_name_tokens)
        batch_name_block_len = pad_batch_data(name_block_len, shape=[-1, self.max_item_len])

        return_list.append(padded_ids)
        return_list.append(mask_ids)
        return_list.append(batch_seq_lens)
        return_list.append(name_len)
        return_list.append(batch_name_pos)
        return_list.append(batch_name_block_len)

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
        record_id_dict[DName.INPUT_IDS] = fields_id[start_index]
        record_id_dict[DName.MASK_IDS] = fields_id[start_index + 1]
        record_id_dict[DName.SEQ_LENS] = fields_id[start_index + 2]
        record_id_dict[DName.NAME_LENS] = fields_id[start_index + 3]
        record_id_dict[DName.NAME_POS] = fields_id[start_index + 4]
        record_id_dict[DName.NAME_TOK_LEN] = fields_id[start_index + 5]

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
        return TEXT_2D_FIELD_NUM

    def _pad_name_block_to_batch(self, insts, data_type="int64", pad_id=0):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        inst_data = np.array([inst + list([[0, 0]] * (self.max_item_len - len(inst))) for inst in insts])
        inst_data = inst_data.astype(data_type).reshape([-1, self.max_item_len, 2])
        return inst_data


if __name__ == "__main__":
    """run some simple test cases"""
    pass

