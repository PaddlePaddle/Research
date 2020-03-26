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

"""one hot feature field reader
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
from text2sql.framework.reader.util_helper import pad_batch_data

from text2sql.datalib import DName

ONE_HOT_FEATURE_FIELD_NUM = 1

@RegisterSet.field_reader.register
class OneHotFeatureFieldReader(BaseFieldReader):

    """One Hot Feature Field Reader. """

    def __init__(self, field_config):
        """init of class

        Args:
            field_config (TYPE): NULL
        """
        super(OneHotFeatureFieldReader, self).__init__(field_config)

        self.paddle_version_code = 1.6
        self._feature_dim = self.field_config.embedding_info["feature_dim"]

    def init_reader(self):
        """ 初始化reader格式
        :return: reader的shape[]、type[]、level[]
        """
        shape = []
        types = []
        levels = []

        if self.field_config.data_type != DataShape.INT:
            raise TypeError("data_type must be int")

        # features
        shape.append([-1, self.field_config.max_seq_len, self._feature_dim])
        levels.append(0)
        types.append('float32')

        return shape, types, levels

    def convert_texts_to_ids(self, batch_text):
        """convert a batch of input text instances to ids

        Args:
            batch_text (list of string): NULL

        Returns: TODO

        Raises: NULL

        """
        max_len = self.field_config.max_seq_len
        batch_fea_list = []
        name_block_len = []
        name_block_begin = []
        name_block_end = []
        for idx_batch, text in enumerate(batch_text):
            fea_str = text.split(' [SEP] ')
            fea_list = [[float(y) for y in x.split(' ')] for x in fea_str]

            # 加上截断策略
            if len(fea_list) > self.field_config.max_seq_len:
                logging.warn('input instance is to long: %s', text)
                fea_list = truncation_words(fea_list, self.field_config.max_seq_len, self.field_config.truncation_type)
            batch_fea_list.append(fea_list)

        return_list = []

        padded = [0] * self._feature_dim
        padded_ids = np.array([inst + list([padded] * (max_len - len(inst))) for inst in batch_fea_list])
        padded_ids = padded_ids.astype('float32').reshape([-1, max_len, self._feature_dim])

        return_list.append(padded_ids)

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
        record_id_dict[C.SRC_IDS] = fields_id[start_index]

        record_dict = {}
        record_dict[C.RECORD_ID] = record_id_dict
        record_dict[C.RECORD_EMB] = None

        return record_dict

    def get_field_length(self):
        """获取当前这个field在进行了序列化之后，在field_id_list中占多少长度
        :return:
        """
        return ONE_HOT_FEATURE_FIELD_NUM


if __name__ == "__main__":
    """run some simple test cases"""
    pass

