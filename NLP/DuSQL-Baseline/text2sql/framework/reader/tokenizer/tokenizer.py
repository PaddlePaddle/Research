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
:py:class:`Tokenizer`
"""
from text2sql.framework.register import RegisterSet
from text2sql.framework.reader.vocabulary import Vocabulary


@RegisterSet.tokenizer.register
class Tokenizer(object):
    """Tokenizer"""

    def __init__(self, vocab_file, split_char=" ", unk_token="[UNK]", params=None):
        """
        :param vocab_file: 词表文件路径
        :param split_char: 明文分隔符，默认是空格
        :param unk_token: unk 对应的token，默认是[UNK]
        :param params: 个别tokenizer自己用到的额外参数，dict类型
        """
        self.vocabulary = Vocabulary(vocab_file, unk_token)
        self.split_char = split_char
        self.unk_token = unk_token
        self.params = params

    def tokenize(self, text):
        """
        :param text:
        :return: tokens, list类型
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens):
        """
        :param tokens:
        :return:
        """
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids):
        """
        :param ids:
        :return:
        """
        raise NotImplementedError

    def covert_id_to_token(self, id):
        """
        :param id:
        :return: token
        """
        return self.vocabulary.covert_id_to_token(id)

    def covert_token_to_id(self, token):
        """
        :param token:
        :return: id
        """
        return self.vocabulary.covert_token_to_id(token)