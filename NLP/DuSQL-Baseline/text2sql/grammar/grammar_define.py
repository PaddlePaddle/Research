#!/usr/bin/env python3
# -*- coding: utf8 -*-
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

"""grammar defination
"""

import sys
import os
import traceback
import logging
from collections import defaultdict

import numpy as np

CONST_KEYWORD = set(['intersect', 'union', 'except',
                 'count', 'sum', 'min', 'max', 'avg', 'none',
                 'between', 'like', 'not_like', 'in', 'not_in',
                 '=', '!=', '<', '>', '<=', '>=', 'and', 'or',
                 '+', '-', '*', '/',
                 'des', 'asc'])
INF = 1e9

class Grammar(object):

    """Grammar Defination"""

    # 可解码为叶子几点的文法数，即Table、Column、Value 3 种
    LEAF_NUM = 3

    TYPE_START = 1
    TYPE_END = 2
    TYPE_CONST = 3
    TYPE_LEAF = 4
    TYPE_MID = 5

    ACTION_NONE = 0
    ACTION_APPLY = 1
    ACTION_SELECT_T = 2
    ACTION_SELECT_C = 3
    ACTION_SELECT_V = 4
    ACTION_STOP = 5

    GMR_T = 'Table'
    GMR_C = 'Column'
    GMR_V = 'Value'
    GMR_NAME_T = -1
    GMR_NAME_C = -1
    GMR_NAME_V = -1
    GMR_NAME_END = -1

    # 最大的table、column、value数量，实际数量batch 内不同样本可能不同
    MAX_TABLE = -1
    MAX_COLUMN = -1
    MAX_VALUE = -1

    def __init__(self, desc_file):
        """init of class

        Args:
            desc_file (TYPE): NULL
        """
        super(Grammar, self).__init__()

        self._desc_file = desc_file
        self._vocab_size = None

        # 初始化为 const keywords 部分，后边读取语法配置词表继续更新
        self.dct_token2id = dict() #[(tok, idx) for idx, tok in enumerate(CONST_KEYWORD)])

        # 模型解码的目标词表空间
        self._lst_gmr_desc = list()
        self._lst_gmr2name = list()

        # grammar token 作为 name 时对应的 ID
        self.dct_token_id2name_id = dict()
        # name id 对应的 grammar id(即grammar.txt中的第n行)
        self._dct_name2gmr = defaultdict(list)

        with open(desc_file) as ifs:
            self._init_from_file(ifs)

        self._lst_gmr_desc, self.gmr_desc_lens, self.max_desc_len = self._generate_gmr_desc(self._lst_gmr_desc)

        self.dct_token_id2name_id[self.LEAF] = len(self.dct_token_id2name_id)
        Grammar.GMR_NAME_T = self.dct_token_id2name_id[self.dct_token2id[self.GMR_T]]
        Grammar.GMR_NAME_C = self.dct_token_id2name_id[self.dct_token2id[self.GMR_C]]
        Grammar.GMR_NAME_V = self.dct_token_id2name_id[self.dct_token2id[self.GMR_V]]
        Grammar.GMR_NAME_END = self.dct_token_id2name_id[self.dct_token2id['END']]

        # shape is: [len(_dct_name2gmr), len(_lst_gmr_desc) - LEAF_NUM]
        self.grammar_mask_matrix = self._init_grammar_mask()

        self._to_numpy()

    def _to_numpy(self):
        """对外的数据结构转为 numpy.ndarray
        Returns: None
        """
        self.gmr_vocab = np.array(range(len(self._lst_gmr_desc) - self.LEAF_NUM)).astype(np.int64)
        # shape = [grammar_size_with_leaf, max_len]
        self.gmr_desc_arr = np.array(self._lst_gmr_desc).astype(np.int64)
        # shape = [grammar_size]
        self.gmr_desc_lens = np.array(self.gmr_desc_lens).astype(np.int64)
        # shape = [name_size - 3, grammar_size]
        self.grammar_mask_matrix = np.array(self.grammar_mask_matrix).astype(np.float32)
        self.gmr2name_arr = self._expand_name_arr(np.int64)
        self.gmr_token2name_arr = self._generate_token_name_arr(np.int64)
        self.gmr2type_arr = self._generate_type_arr(np.int64)
        self.gmr_token2action_arr, self.gmr_name2action = self._generate_action_arr(np.int64)

    @property
    def grammar_size(self):
        """read property of static/non-leaf grammar size"""
        return len(self.gmr_vocab)

    @property
    def vocab_size(self):
        """read property of vocab_size"""
        if self._vocab_size is None:
            self._vocab_size = self.grammar_size + self.MAX_TABLE + self.MAX_COLUMN + self.MAX_VALUE
        return self._vocab_size

    @property
    def grammar_size_with_leaf(self):
        """read property of grammar size with 3 leaf ones"""
        return self.grammar_size + self.LEAF_NUM

    @property
    def name_size(self):
        """read property of vocab_size"""
        return len(self._dct_name2gmr)

    def _init_from_file(self, ifs):
        """init grammar desc and ids from file(ifs, Input File Stream)

        Args:
            ifs (file stream): NULL

        Returns: TODO

        Raises: NULL

        """
        lines = [x.strip() for x in ifs]
        # 过滤空行和注释行（#号开始）
        lines = filter(lambda x: not(len(x) == 0 or x.startswith('#')), lines)

        for gmr_id, line in enumerate(lines):
            gmr_rule_tokens = [x for x in line.split(' ') if x not in CONST_KEYWORD]
            # 更新词表
            self._update_token_vocab(gmr_rule_tokens)

            lst_token_ids = list(self.dct_token2id[x] for x in gmr_rule_tokens)

            gmr_name_id = lst_token_ids[0]
            gmr_desc_ids = lst_token_ids[1:]
            self._lst_gmr_desc.append(gmr_desc_ids)

            # updates:
            # 1. self.dct_token_id2name_id
            # 2. self._dct_name2gmr
            # 3. self._lst_gmr2name
            self._update_gmr_name(gmr_name_id, gmr_id)

    def _update_token_vocab(self, gmr_rule_tokens):
        """更新词表：补充不在词表中的新 token

        Args:
            gmr_rule_tokens (list): NULL

        Returns: None

        Raises: NULL

        """
        for token in gmr_rule_tokens:
            if token not in self.dct_token2id:
                self.dct_token2id[token] = len(self.dct_token2id)

    def _update_gmr_name(self, gmr_token_id, gmr_id):
        """update grammar name name-to-id dict, and name-id-to-grammar-id dict.

        Args:
            gmr_token_id (TYPE): NULL
            gmr_id (TYPE): NULL

        Returns: None

        Raises: NULL

        """
        if gmr_token_id not in self.dct_token_id2name_id:
            self.dct_token_id2name_id[gmr_token_id] = len(self.dct_token_id2name_id)
        nameid = self.dct_token_id2name_id[gmr_token_id]
        self._dct_name2gmr[nameid].append(gmr_id)
        self._lst_gmr2name.append(nameid)

    def _init_grammar_mask(self):
        """初始化静态文法的mask矩阵，即除去解码为叶子节点的部分。
        解码为叶子节点的Table等需要运行时动态创建mask矩阵。

        Returns: list of list

        Raises: NULL

        """
        mask_matrix = []
        static_grammar_len = len(self._lst_gmr_desc) - self.LEAF_NUM
        for nameid, valid_gmr_ids in sorted(self._dct_name2gmr.items())[:-self.LEAF_NUM]:
            mask = [-INF] * static_grammar_len
            for gmr_id in valid_gmr_ids:
                mask[gmr_id] = 0.
            mask_matrix.append(mask)
        return mask_matrix

    def _generate_gmr_desc(self, lst_2d, padding=0):
        """convert 2d list to 2d np.ndarray

        Args:
            lst_2d (TYPE): NULL
            padding (TYPE): Default is 0

        Returns: (lst_2d, lens)

        Raises: NULL

        """
        ##lst_2d = lst_2d[:-self.LEAF_NUM] + [list()] * (self.max_table + self.max_column + self.max_value)
        lens = [len(x) for x in lst_2d]
        max_len = max(lens)
        for lst in lst_2d:
            lst.extend([padding] * (max_len - len(lst)))

        return lst_2d, [[x] for x in lens], max_len

    def _expand_name_arr(self, dtype=np.int64):
        """expand token id to name id array, according to max_table, max_column, max_value

        Args:
            dtype (np.dtype): output dtype

        Returns: TODO

        Raises: NULL
        """
        lst_gmr2name_expanded = self._lst_gmr2name[:-self.LEAF_NUM] + \
                                [self.GMR_NAME_T] * self.MAX_TABLE + \
                                [self.GMR_NAME_C] * self.MAX_COLUMN + \
                                [self.GMR_NAME_V] * self.MAX_VALUE
        return np.array(lst_gmr2name_expanded).astype(dtype)

    def _generate_token_name_arr(self, dtype=np.int64):
        """generate token id to name id array

        Args:
            dtype (np.dtype): output dtype

        Returns: TODO

        Raises: NULL
        """
        lst_token2name = [self.GMR_NAME_END] * len(self.dct_token2id)
        for tid, nid in self.dct_token_id2name_id.items():
            lst_token2name[tid] = nid
        return np.array(lst_token2name).astype(dtype)

    def _generate_type_arr(self, dtype=np.int64):
        """generate token id to type id array.

        Args:
            dtype (np.dtype): output dtype

        Returns: TODO

        Raises: NULL
        """
        lst_gmr2type_no_leaf = [self.TYPE_MID] * self.grammar_size
        lst_gmr2type_no_leaf[self.END] = self.TYPE_END
        lst_gmr2type_no_leaf[self.START] = self.TYPE_START
        #const_token_ids = [self.dct_token2id[x] for x in CONST_KEYWORD]
        #for tid in const_token_ids:
        #    lst_gmr2type_no_leaf[tid] = self.TYPE_CONST

        lst_gmr2type = lst_gmr2type_no_leaf + [self.TYPE_LEAF] * (self.MAX_TABLE + self.MAX_COLUMN + self.MAX_VALUE)
        return np.array(lst_gmr2type).astype(dtype)

    def _generate_action_arr(self, dtype=np.int64):
        """generate token id to action id array.

        Args:
            dtype (np.dtype): output dtype

        Returns: TODO

        Raises: NULL
        """
        gmr_token2action = np.array([self.ACTION_APPLY] * len(self.dct_token2id))
        gmr_token2action[self.Table] = self.ACTION_SELECT_T
        gmr_token2action[self.Column] = self.ACTION_SELECT_C
        gmr_token2action[self.Value] = self.ACTION_SELECT_V
        gmr_token2action[self.END_TOKEN] = self.ACTION_STOP

        gmr_name2action = np.array([self.ACTION_APPLY] * self.name_size)
        gmr_name2action[self.GMR_NAME_T] = self.ACTION_SELECT_T
        gmr_name2action[self.GMR_NAME_C] = self.ACTION_SELECT_C
        gmr_name2action[self.GMR_NAME_V] = self.ACTION_SELECT_V
        gmr_name2action[self.GMR_NAME_END] = self.ACTION_STOP
        return gmr_token2action.astype(dtype), gmr_name2action.astype(dtype)

    def __getattr__(self, token):
        """wrapper of get grammar token id from self.dct_token2id

        Args:
            token (TYPE): NULL

        Returns: TODO

        Raises: NULL

        """
        if token == 'END':
            return self.grammar_size - 1
        if token in self.dct_token2id:
            return self.dct_token2id[token]
        else:
            raise AttributeError("'Grammar' object has no attribute '%s'" % (token))


if __name__ == '__main__':
    import json
    Grammar.MAX_TABLE = 12
    Grammar.MAX_COLUMN = 40
    Grammar.MAX_VALUE = 60
    gmr = Grammar('conf/grammar_no_value.txt')
    print("grammar desc:")
    print(gmr._lst_gmr_desc)
    #print(gmr.gmr_desc_arr)
    print('grammar to name id:')
    print(gmr.gmr2name_arr)
    print('grammar_mask_matrix.shape:', gmr.grammar_mask_matrix.shape)
    print("grammar_mask_matrix:")
    print(gmr.grammar_mask_matrix)
    print('START:', gmr.START, 'END:', gmr.END)
    print('name_size:', gmr.name_size)
    print('grammar_size:', gmr.grammar_size)
    print('vocab_size:', gmr.vocab_size)
    print('token id:')
    print(json.dumps(gmr.dct_token2id, indent=4))
    print('grammar token id to name id:')
    print(sorted(gmr.dct_token_id2name_id.items()))
    print("gmr.gmr2type_arr:")
    print(gmr.gmr2type_arr.tolist())
    print("gmr.gmr_name2action:")
    print(gmr.gmr_name2action)
    print("gmr.gmr_token2action_arr:")
    print(gmr.gmr_token2action_arr)
    print("gmr.gmr_token2name_arr:")
    print(gmr.gmr_token2name_arr)

    print('-' * 100)
    print("test get attr:")
    print(gmr.SQL)
    try:
        print(gmr.ABCD)   # will raise AttributeError
    except AttributeError as ae:
        print(str(ae))

