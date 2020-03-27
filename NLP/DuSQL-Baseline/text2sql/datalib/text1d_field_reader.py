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

"""token sequence field reader
"""

from text2sql.framework.register import RegisterSet
from text2sql.framework.rule import InstanceName as C
from text2sql.framework.rule import DataShape
from text2sql.framework.reader.field_reader.base_field_reader import BaseFieldReader
from text2sql.framework.modules.token_embedding.custom_fluid_embedding import CustomFluidTokenEmbedding
from text2sql.framework.utils.util_helper import truncation_words

from text2sql.utils import pad_batch_data
from text2sql.datalib import DName

TEXT_1D_FIELD_NUM = 3

@RegisterSet.field_reader.register
class Text1DFieldReader(BaseFieldReader):
    """通用文本（string）类型的field_reader,文本处理规则是，文本类型的数据会自动添加padding和mask，并返回length
    """
    def __init__(self, field_config):
        """
        :param field_config:
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

        if self.field_config.embedding_info and self.field_config.embedding_info["use_reader_emb"]:
            self.token_embedding = CustomFluidTokenEmbedding(emb_dim=self.field_config.embedding_info["emb_dim"],
                                                       vocab_size=self.tokenizer.vocabulary.get_vocab_size())

    def init_reader(self):
        """ 初始化reader格式
        :return: reader的shape[]、type[]、level[]
        """
        shape = []
        types = []
        levels = []
        if self.field_config.data_type == DataShape.STRING:
            """src_ids"""
            shape.append([-1, self.field_config.max_seq_len, 1])
            levels.append(0)
            types.append('int64')
        else:
            raise TypeError("Text1DFieldReader's data_type must be string")

        """mask_ids"""
        shape.append([-1, self.field_config.max_seq_len, 1])
        levels.append(0)
        types.append('float32')
        """seq_length"""
        shape.append([-1])
        levels.append(0)
        types.append('int64')

        return shape, types, levels

    def convert_texts_to_ids(self, batch_text):
        """将一个batch的明文text转成id
        :param batch_text:
        :return:
        """
        max_len = self.field_config.max_seq_len

        src_ids = []
        for text in batch_text:
            if self.field_config.need_convert:
                tokens = self.tokenizer.tokenize(text)
                src_id = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                src_id = text.split(" ")

            # 加上截断策略
            if len(src_id) > self.field_config.max_seq_len:
                src_id = truncation_words(src_id, self.field_config.max_seq_len, self.field_config.truncation_type)
            src_ids.append(src_id)

        return_list = []
        padded_ids, mask_ids, batch_seq_lens = pad_batch_data(src_ids,
                                                              max_len=self.field_config.max_seq_len,
                                                              pad_idx=self.field_config.padding_id,
                                                              return_input_mask=True,
                                                              return_seq_lens=True,
                                                              paddle_version_code=self.paddle_version_code)
        return_list.append(padded_ids)
        return_list.append(mask_ids)
        return_list.append(batch_seq_lens)

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
        return TEXT_1D_FIELD_NUM


