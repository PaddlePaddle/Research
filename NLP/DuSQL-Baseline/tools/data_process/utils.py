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

""" propress utils
"""

import os
import json
import re
from collections import OrderedDict

import jieba

AGG = ['平均', '总和', '一共', '最']
CONST_COLUMN_DICT = {'TIME_NOW': '当前时间'}

def segword(sentence, seg_to_char=False):
    """wrapper of word segmentation

    Args:
        sentence (TYPE): NULL
        seg_to_char (TYPE): Default is False

    Returns: TODO

    Raises: NULL
    """
    def is_en_alnum(s):
        """check if s is English album or number
        Args:
            s (str): NULL
        Returns: bool
        """
        return s.isalnum() and ord(s) < 128

    if len(sentence) == 0:
        return []

    if not seg_to_char:
        return list(jieba.cut(sentence))

    lst_result = [sentence[0]]
    last_is_ascii = lst_result[-1].isalnum()
    for char in sentence[1:]:
        if char == ' ':
            last_is_ascii = False
            continue

        if is_en_alnum(char) and last_is_ascii:
            lst_result[-1] += char
            continue

        if is_en_alnum(char):
            last_is_ascii = True
        else:
            last_is_ascii = False

        lst_result.append(char)

    return lst_result


def load_dataset(data_path, table_path, cell_path, max_table):
    """load dataset
    Args:
        data_path: train data path
        table_path: db path
        cell_path: db cells path
        max_table: max table setting of dataset
    Returns: TODO
    Rasises: NULL
    """
    with open(data_path, 'r', encoding='utf8') as f:
        datas = json.load(f)
    with open(table_path, 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    with open(cell_path, 'r', encoding='utf8') as f:
        cells = json.load(f)

    dct_tables = {}
    for i in range(len(table_datas)):
        table = table_datas[i]
        tmp_col = OrderedDict()
        for tid, cname in table['column_names']:
            if cname not in tmp_col:
                tmp_col[cname] = [0.0] * max_table
            if tid < 0:
                for i in range(len(table['table_names'])):
                    tmp_col[cname][i] = 1.0
                    continue
            tmp_col[cname][tid] = 1.0
        table['column_set'] = list(tmp_col.keys())
        table['column_set_tables'] = list(tmp_col.values())
        db_name = table['db_id']
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['column_table'] = [col[0] for col in table['column_names']]
        dct_tables[db_name] = table
        for const_col in CONST_COLUMN_DICT.values():
            table['column_set'].append(const_col)

    dct_cells = {}
    for db_cell in cells:
        db_id = db_cell['db_id']
        dct_cells[db_id] = db_cell

    for d in datas:
        d['question'] = d['question'].replace('\n', '').strip()
        d['column_set'] = dct_tables[d['db_id']]['column_set']
        d['column_names'] = dct_tables[d['db_id']]['schema_content']
        d['table_names'] = dct_tables[d['db_id']]['table_names']
        d['column_table'] = dct_tables[d['db_id']]['column_table']
        d['column_set_tables'] = dct_tables[d['db_id']]['column_set_tables']
        d['question_tokens'] = [x.lower() for x in jieba.cut(d['question']) if x.strip() != '']

    return datas, dct_tables, dct_cells


def group_header(toks, idx, header_toks):
    """
    match question literal group header
    """
    num_toks = len(toks)
    for idx_end in reversed(range(idx + 1, num_toks + 1)):
        sub_toks = toks[idx: idx_end]
        sub_toks = "".join(sub_toks).replace(" ", "")
        if sub_toks in header_toks:
            return idx_end, sub_toks
    return idx, None


def fully_part_header(toks, idx, header_toks):
    """
    match question literal fully part header
    """
    num_toks = len(toks)
    for idx_end in reversed(range(idx + 1, num_toks + 1)):
        sub_toks = toks[idx: idx_end]
        if len(sub_toks) > 1:
            sub_toks = "".join(sub_toks).replace(" ", "")
            if sub_toks in header_toks:
                return idx_end, sub_toks
    return idx, None


def partial_header(toks, idx, header_toks):
    """ 
    match question literal partial header
    """
    def check_in(list_one, list_two):
        """check in or not """
        list_one="".join(list_one)
        if list_one in list_two[0]:
            return True
        return False
    for idx_end in reversed(range(idx + 1, len(toks))):
        sub_toks = toks[idx: min(idx_end, len(toks))]
        if len(sub_toks) > 1:
            flag_count = 0
            tmp_heads = None
            for heads in header_toks:
                if check_in(sub_toks, heads):
                    flag_count += 1
                    tmp_heads = heads
            if flag_count == 1:
                return idx_end, tmp_heads
 
    return idx, None


def group_digital(toks, idx):
    """
    is digital or not
    """
    test = toks[idx].replace(':', '')
    test = test.replace('.', '')
    return test.isdigit()


def extract_gold_value(dct_sql):
    """extract gold value from sql

    Args:
        dct_sql (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    value_set = set()
    nest_sql = []

    conditions = [cond for idx, cond in enumerate(dct_sql['where']) if idx % 2 == 0] + \
                 [cond for idx, cond in enumerate(dct_sql['having']) if idx % 2 == 0]
    for cond in conditions:
        if type(cond[3]) is dict:
            nest_sql.append(cond[3])
            continue
        if type(cond[4]) is dict:
            nest_sql.append(cond[4])
            continue
        if cond[3] is not None:
            value_set.add(str(cond[3]))
        if cond[4] is not None:
            value_set.add(str(cond[4]))
    if dct_sql.get('limit', None) is not None:
        value_set.add(str(dct_sql['limit']))

    nest_sql += [table for table_type, table in dct_sql['from']['table_units'] if table_type == 'sql']
    for sql in nest_sql:
        value_set.update(extract_gold_value(sql))

    if dct_sql['intersect'] is not None:
        value_set.update(extract_gold_value(dct_sql['intersect']))
    if dct_sql['union'] is not None:
        value_set.update(extract_gold_value(dct_sql['union']))
    if dct_sql['except'] is not None:
        value_set.update(extract_gold_value(dct_sql['except']))

    return list(value_set)


def match_score(sentence, token_set, base=0):
    """TODO: Docstring for match_score.

    Args:
        sentence (TYPE): NULL
        token_set (TYPE): NULL
        base (TYPE): Default is 0

    Returns: TODO

    Raises: NULL
    """
    sentence_seg = list(jieba.cut(sentence))
    match_cnt = len(set(sentence_seg) & token_set)
    return match_cnt / len(sentence_seg) + base if len(sentence_seg) > 0 else base


def is_simple_float(token):
    """simple float xx.yy, not sientific notation

    Args:
        token (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    if token.count('.') != 1:
        return False
    int_part, float_part = token.split('.')
    return int_part.isdigit() and float_part.isdigit()


def is_int_float(val):
    """'int_float' mean a float number which float part is zero

    Args:
        val (float): NULL

    Returns: TODO

    Raises: NULL
    """
    return int(val) * 10000 == int(val * 10000)


if __name__ == "__main__":
    """run some simple test cases"""
    s = '你好，world.'
    for char in s:
        print(char, char.isalnum(), ord(char))
    print(segword(s, seg_to_char=False))
    print(segword(s, seg_to_char=True))

