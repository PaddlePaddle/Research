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

"""grammar utils
"""

import os
import json
import argparse
import logging
import re as regex
from grammar import Superlative
from grammar import Select
from grammar import Order
from grammar import SingleSQL
from grammar import Filter
from grammar import Agg
from grammar import NumA
from grammar import Column
from grammar import Table
from grammar import SQL
from grammar import MathAgg
from grammar import Value


def _id(rule):
    """get id in rule, like 7 in SQL(7), or 8 in Table(8)

    Args:
        rule (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    id_str = rule[:-1].rsplit('(')[1]
    return int(id_str)


def load_grammar_vocab(grammar_file, max_len):
    """load grammar vocab from file

    Args:
        grammar_file (TYPE): NULL
        max_len (tuple): (max_table, max_column, max_value)

    Returns: TODO

    Raises: NULL
    """
    with open(grammar_file, 'r', encoding='utf8') as ifs:
        lst_grammar = [x for x in ifs if not x.strip().startswith('#')]
    grammar_vocab = []
    curr_idx = -1
    last = None
    for grammar in lst_grammar:
        name = grammar.split(' ')[0]
        if name in ('Table', 'Column', 'Value'):
            continue
        if name != last:
            curr_idx = 0
            last = name
        grammar_vocab.append('%s(%d)' % (name, curr_idx))
        curr_idx += 1
    max_table, max_column, max_value = max_len
    grammar_vocab += ['Table(%d)' % (idx) for idx in range(max_table)]
    grammar_vocab += ['Column(%d)' % (idx) for idx in range(max_column)]
    grammar_vocab += ['Value(%d)' % (idx) for idx in range(max_value)]
    return grammar_vocab


def load_datasets(data_path, table_path, grammar_path, rule_path, max_len):
    """
    load data
    Args:
        data_path:
        table_path:
        grammar_path:
    Return:
        TODO
    Raise:
        NULL
    """
    with open(data_path, 'r', encoding='utf8') as ifs:
        lst_data = json.load(ifs)
    with open(table_path, 'r', encoding='utf8') as ifs:
        table_datas = json.load(ifs)

    dct_schema = dict()
    for table in table_datas:
        dct_schema[table['db_id']] = table

    with open(rule_path, 'r', encoding='utf8') as ifs:
        lst_rules = [x.strip() for x in ifs]

    if len(lst_data) != len(lst_rules):
        raise IOError('input data len(%d) != grammar rule len(%d)' % (len(lst_data), len(lst_rules)))
    grammar_vocab = load_grammar_vocab(grammar_path, max_len)
    for dct_data, pred_rule_id in zip(lst_data, lst_rules):
        dct_data['predicted_rule' ] = pred_rule_id

        if pred_rule_id.strip() == '':
            pred_rule_id = '8 14 15 41 45 76 61 48 73 61'
        predicted_rule_tokens = [grammar_vocab[int(x)] for x in pred_rule_id.split(' ')]
        dct_data['predicted_rule_tokens'] = ' '.join(predicted_rule_tokens)

    return lst_data, dct_schema


def partial_match(query, table_name):
    """
    query partial match table_name
    Args:
        query: question
        table_name: table_name
    Return:[True/False]
    Raise: Null
    """
    query = [x for x in query]
    table_name = [x for x in table_name]
    if query in table_name:
        return True
    return False


def is_partial_match(query, table_names):
    """
    query is partial match table_name
    Args:
        query: question
        table_name: table_names
    Return:[result/False]
    Raise: Null
    """
    query = query
    table_names = [[x for x in names.split(' ') ] for names in table_names]
    same_count = 0
    result = None
    for names in table_names:
        if query in names:
            same_count += 1
            result = names
    return result if same_count == 1 else False


def multi_option(question, idx_q, names, N):
    """
    multi_option
    Args:
        question: question
        idx_q:
        names:
        N
    Return:[re/False]
    Raise: Null
    """

    for i in range(idx_q + 1, idx_q + N + 1):
        if i < len(question):
            re = is_partial_match(question[i][0], names)
            if re is not False:
                return re
    return None


def multi_equal(question, idx_q, names, N):
    """
    multi equal
    Args:
        question: question
        idx_q: 
        names:
        N:
    Return:[i/False]
    Raise: Null
    """

    for i in range(idx_q + 1, idx_q + N + 1):
        if i < len(question):
            if question[i] == names:
                return i
    return None


def random_choice(question_arg, question_arg_type, names, ground_col_labels, idx_q, N, origin_name):
    """ randoom choice"""
    # first try if there are other table
    for t_ind, t_val in enumerate(question_arg_type):
        if t_val == ['table']:
            return names[origin_name.index(question_arg[t_ind])]
    for i in range(idx_q + 1, idx_q + N + 1):
        if i < len(question_arg):
            if len(ground_col_labels) == 0:
                for n in names:
                    if partial_match(question_arg[i][0], n) is True:
                        return n
            else:
                for n_id, n in enumerate(names):
                    if n_id in ground_col_labels and partial_match(question_arg[i][0], n) is True:
                        return n
    if len(ground_col_labels) > 0:
        return names[ground_col_labels[0]]
    else:
        return names[0]


def find_table(cur_table, origin_table_names, question_arg_type, question_arg):
    """find table"""
    h_table = None
    for i in range(len(question_arg_type))[::-1]:
        if question_arg_type[i] == ['table']:
            h_table = question_arg[i]
            h_table = origin_table_names.index(h_table)
            if h_table != cur_table:
                break
    if h_table != cur_table:
        return h_table

    # find partial
    for i in range(len(question_arg_type))[::-1]:
        if question_arg_type[i] == ['NONE']:
            for t_id, table_name in enumerate(origin_table_names):
                if partial_match(question_arg[i], table_name) is True and t_id != h_table:
                    return t_id

    # random return
    for i in range(len(question_arg_type))[::-1]:
        if question_arg_type[i] == ['table']:
            h_table = question_arg[i]
            h_table = origin_table_names.index(h_table)
            return h_table

    return cur_table


def alter_not_in(datas, schemas):
    """Filter not_in A SingleSQL"""
    for d in datas:
        if 'label_str' not in d or 'Filter(19)' not in d['label_str']:
            continue
        current_table = schemas[d['db_id']]
        current_table['schema_content_clean'] = [x[1] for x in current_table['column_names']]
        origin_table_names = [[x.lower() for x in names] for names in d['table_names']]
        #origin_table_names = [[x.lower() for x in names.split(' ')] for names in d['table_names']]
        question_arg_type = d['question_literal_type']
        question_arg = d['question_tokens']
        pred_label = d['label_str'].split(' ')

        # get potiantial table
        f_19_idx = pred_label.index('Filter(19)')
        cur_table = _id(pred_label[f_19_idx - 1])

        h_table = find_table(cur_table, origin_table_names, question_arg_type, question_arg)

        column_names = [''.join(x) for x in d['column_names']]
        for label_id, label_val in enumerate(pred_label):
            if label_val != 'Filter(19)':
                continue
            for primary in current_table['primary_keys']:
                if current_table['column_names'][primary][0] == _id(pred_label[label_id - 1]):
                    prim_key = current_table['schema_content_clean'][primary]
                    pred_label[label_id + 2] = 'Column(%d)' % (column_names.index(prim_key))
                    break
            
            for pair in current_table['foreign_keys']:
                curr_key = current_table['schema_content_clean'][pair[0]]
                foreign_key = current_table['schema_content_clean'][pair[1]]
                if current_table['column_names'][pair[0]][0] == h_table and \
                        column_names.index(foreign_key) == _id(pred_label[label_id + 2]):
                    pred_label[label_id + 8] = 'Column(%d)' % (column_names.index(curr_key))
                    pred_label[label_id + 9] = 'Table(' + str(h_table) + ')'
                    break
                elif current_table['column_names'][pair[1]][0] == h_table and \
                        column_names.index(curr_key) == _id(pred_label[label_id + 2]):
                    pred_label[label_id + 8] = 'Column(%d)' % (column_names.index(foreign_key))
                    pred_label[label_id + 9] = 'Table(' + str(h_table) + ')'
                    break
            pred_label[label_id + 3] = pred_label[label_id - 1]

        d['label_str'] = " ".join(pred_label)


