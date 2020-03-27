# -*- coding: utf-8 -*-
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
"""
:py:class:`CustomFluidTokenEmbedding`
"""
from paddle import fluid

from text2sql.framework.rule import InstanceName
from text2sql.framework.modules.token_embedding.base_token_embedding import BaseTokenEmbedding


class CustomFluidTokenEmbedding(BaseTokenEmbedding):
    """CustomFluidTokenEmbedding: 使用paddle.fluid 的api实现的embedding，随机初始化，训练过程中不断finetune
    """
    def __init__(self, emb_dim, vocab_size):
        BaseTokenEmbedding.__init__(self, emb_dim, vocab_size)
        self.params_name = None

    def build(self):
        """
        添加一些自顶一个初始化信息，如参数名称
        :return:
        """
        pass

    def get_token_embedding(self, tokens_dict):
        """
        :param tokens_dict:
        :return:
        """
        tokens = tokens_dict[InstanceName.SRC_IDS]
        tokens_length = tokens_dict[InstanceName.SEQ_LENS]
        unpad_data = fluid.layers.sequence_unpad(tokens, length=tokens_length)
        emb = fluid.layers.embedding(input=unpad_data, size=[self.vocab_size, self.emb_dim])
        emb_dict = {
            InstanceName.SEQUENCE_EMB: emb,
            InstanceName.POOLED_EMB: None
        }
        return emb_dict

    def get_output_dim(self):
        """
        :return:
        """
        return self.emb_dim
