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

"""data process for text2sql model.
"""

import sys
import os
import traceback
import logging
import argparse
import json
import re
from collections import defaultdict
import numpy as np
import jieba

import utils
import sql2grammar

g_max_values = 60

jieba_cut = lambda sent: [x for x in jieba.cut(sent) if x.strip() != '']

def set_p_match_fea(q_tokens, names, pos, fea_matrix):
    """set partial match feature

    Args:
        q_tokens (TYPE): NULL
        names (TYPE): NULL
        pos (TYPE): NULL
        fea_matrix (TYPE): [out]

    Returns: TODO

    Raises: NULL
    """
    for idx_name, name in enumerate(names):
        for q_tok in q_tokens:
            if q_tok in name:
                fea_matrix[idx_name][pos] += 1


def schema_linking(
        question_arg, question_arg_type, one_hot_type, col_set_type, tab_set_type, column_set, table_names, sql):
    """
    schema_linking
    :return [one_hot_type,column_set]  
    """
    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            try:
                question_table = ''.join(question_arg[count_q]).strip()
                tab_set_type[table_names.index(question_table)][0] = 5
            except:
                raise RuntimeError("not in table names")
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            try:
                question_column = ''.join(question_arg[count_q]).strip()
                col_set_type[column_set.index(question_column)][0] = 5
            except:
                raise RuntimeError("not in col set")
        elif t == 'agg':
            one_hot_type[count_q][2] = 1
        elif t == 'MORE':
            one_hot_type[count_q][3] = 1
        elif t == 'MOST':
            one_hot_type[count_q][4] = 1
        elif t == 'value':
            one_hot_type[count_q][5] = 1
        elif len(t_q) == 1:
            for col_probase in t_q:
                if col_probase == 'asd':
                    continue
                try:
                    col_set_type[sql['column_names'].index(col_probase)][2] = 5
                    question_arg[count_q] = ['value'] + question_arg[count_q]
                except:
                    raise RuntimeError('not in col')
                one_hot_type[count_q][5] = 1
        else:
            for col_probase in t_q:
                if col_probase == 'asd':
                    continue
                col_set_type[sql['column_names'].index(col_probase)][3] += 1


def process(instance):
    """
    use jieba to participle
    Args:
        instance : question's json data
    return:
        TODO
    raise:
        NULL
    """
    process_dict = {}

    process_dict['column_set'] = [jieba_cut(x) for x in instance['column_set']]
    process_dict['q_iter_small'] = instance['question_tokens']
    process_dict['table_names'] = [jieba_cut(x) for x in instance['table_names']]
    process_dict['column_names'] = [jieba_cut(x) for x in instance['column_names']]

    return process_dict


def process_cells(dct_cells, data, question_tokens, is_train):
    """
    process cells 
    dct_cells: db table's cell: {db_id: cells}
    data : train data
    is_train: if train is True else is False
    :return: [value ,value_features,val_col_tab]
    """
    cell_columns=[]
    dct_value_score = defaultdict(int)

    if is_train:
        gold_values = utils.extract_gold_value(data['sql'])
        for idx, val in enumerate(gold_values):
            if utils.is_simple_float(val) and utils.is_int_float(float(val)):
                gold_values[idx] = str(int(float(val)))
        dct_value_score.update((v, 100) for v in set(gold_values))

    curr_cells = dct_cells[data['db_id']]
    q_tok_set = set(question_tokens)
    for table_name, table in curr_cells['tables'].items():
        table_score=  utils.match_score(table_name, q_tok_set)
        col_dtypes = table['type']
        cols = table['header']
        col_scores = [utils.match_score(c, q_tok_set, base=table_score) for c in cols]
        for row in table['cell']:
            for one_cell, col_score, col_dtype in zip(row, col_scores, col_dtypes):
                if col_dtype in ('number', 'time') or one_cell == "":
                    continue
                dct_value_score[one_cell] += utils.match_score(one_cell, q_tok_set, base=col_score)

    for idx, word in enumerate(question_tokens[:-1]):
        dct_value_score[word] += 0.1
        dct_value_score[word + question_tokens[idx + 1]] += 0.1
    dct_value_score[question_tokens[-1]] += 0.1
    if len(dct_value_score) <= g_max_values:
        values = list(sorted(dct_value_score.keys()))
    else:
        values = [x[0] for x in sorted(dct_value_score.items(), key=lambda x: x[1], reverse=True)[:g_max_values]]

    value_features = np.zeros((len(values), 2), dtype=np.int) # EM, PM
    set_p_match_fea(data['question_tokens'], values, 1, value_features)

    idx = 0
    values_set = set(values)
    question_toks = data['question_tokens']
    while idx < len(question_toks):
        end_idx, cellvalue = utils.group_header(question_toks, idx, values_set)
        if cellvalue:
            idx = end_idx
            for i, v in enumerate(values):
                if cellvalue == v:
                    value_features[i][0] = 5
        else:
            idx += 1

    return values, value_features, []


def set_question_literal_info(question_toks, column_names, table_names, question_literal, q_literal_type):
    """TODO: Docstring for set_question_literal_info.

    Args:
        question_toks (TYPE): NULL
        column_names (TYPE): NULL
        table_names (TYPE): NULL
        question_literal (TYPE): [out]
        q_literal_type (TYPE): [out]

    Returns: TODO

    Raises: NULL
    """
    idx = 0
    while idx < len(question_toks):
        # fully header
        end_idx, header = utils.fully_part_header(question_toks, idx, column_names)
        if header:
            question_literal.append(question_toks[idx: end_idx])
            q_literal_type.append(["col"])
            idx = end_idx
            continue

        # check for table
        end_idx, tname = utils.group_header(question_toks, idx, table_names)
        if tname:
            question_literal.append(question_toks[idx: end_idx])
            q_literal_type.append(["table"])
            idx = end_idx
            continue

        # check for column
        end_idx, header = utils.group_header(question_toks, idx, column_names)
        if header:
            question_literal.append(question_toks[idx: end_idx])
            q_literal_type.append(["col"])
            idx = end_idx
            continue

        if utils.group_digital(question_toks, idx):
            question_literal.append(question_toks[idx: idx + 1])
            q_literal_type.append(["value"])
            idx += 1
            continue

        question_literal.append([question_toks[idx]])
        q_literal_type.append(['NONE'])
        idx += 1


def expand_q_features(question_literal, q_literal_features):
    """expand q features

    Args:
        question_literal (TYPE): NULL
        q_literal_features (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    lst_result = []
    for tokens, feature in zip(question_literal, q_literal_features):
        lst_result.extend([feature] * len(tokens))
    return lst_result


def preprocess(lst_data, dct_cells, sql2grammar_parser, is_train):
    """data preprocess for model training

    Args:
        lst_data (TYPE): NULL
        dct_cells (TYPE): NULL
        sql2grammar_parser (TYPE): NULL
        is_train (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    for entry in lst_data:
        question_toks = [x for x in entry['question_tokens']]
        table_names = entry['table_names']
        column_names = entry['column_names']

        question_literal = []
        q_literal_type = []
        set_question_literal_info(question_toks, column_names, table_names, question_literal, q_literal_type)

        entry['question_literal'] = question_literal
        entry['question_literal_type'] = q_literal_type 

        # [table,col,agg,more,most,value]
        entry['question_features'] = np.zeros((len(entry['question_literal_type']), 6), dtype=np.int)
        # [Exact Match, Partial Match, Value Exact Match, Value Partial Match]
        entry['column_features'] = np.zeros((len(entry['column_set']), 4), dtype=np.int)
        # [Exact Match, Partial Match]
        entry['table_features'] = np.zeros((len(entry['table_names']), 2), dtype=np.int)
        process_dict = process(entry)
        set_p_match_fea(process_dict['q_iter_small'], process_dict['table_names'], 1, entry['table_features'])
        set_p_match_fea(process_dict['q_iter_small'], process_dict['column_set'], 1, entry['column_features'])

        values, value_features, value_col_tab = process_cells(dct_cells, entry, entry['question_tokens'], is_train)

        entry['values'] = [jieba_cut(x) for x in values]
        entry['value_features'] = value_features.tolist() 
        entry['value_col_tab'] = value_col_tab

        schema_linking(entry['question_literal'], entry['question_literal_type'], entry['question_features'], 
                       entry['column_features'], entry['table_features'], entry['column_set'], entry['table_names'],
                       entry)

        entry['question_tokens_original'] = entry['question_tokens']
        entry['question_tokens'] = entry['question_literal']
        entry['question_features'] = entry['question_features'].tolist()
        entry['table_names'] = process_dict['table_names']
        entry['column_names'] = process_dict['column_names']
        entry['column_features'] = entry['column_features'].tolist()
        entry['column_tables'] = entry['column_set_tables']
        entry['table_features'] = entry['table_features'].tolist()
        entry['query_toks_no_value'] = [x for x in jieba_cut(entry['sql_query']) if x != " "]

        if 'sql_query' in entry and entry['sql_query'].strip() != '':
            rule_obj = sql2grammar_parser.parse(entry)
            rule_ids = sql2grammar_parser.grammar2id(rule_obj)
            entry['label_str'] = ' '.join([str(x) for x in rule_obj])
            entry['label'] = ' '.join([str(x) for x in rule_ids])

        entry['column_names_original'] = entry['column_names']
        entry['column_names'] = process_dict['column_set']

        del entry['column_set']
        del entry['column_set_tables']
        del entry['column_table']
        del entry['question_literal']

    return lst_data


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data-path', required=True, help='dataset')
    arg_parser.add_argument('-f', '--flag', default='false', help='train data or not')
    arg_parser.add_argument('-t', '--table-path', required=True, help='table dataset')
    arg_parser.add_argument('-c', '--cell-path', required=True, help='cell dataset')
    arg_parser.add_argument('-g', '--grammar-path', required=True, help='grammar file path')
    arg_parser.add_argument('-m', '--max-len', required=False, nargs=3, type=int, default=(12, 40, g_max_values),
                                               help='max len of table and column. like "-m 20 60 100"')
    arg_parser.add_argument('-o', '--output', required=True, help='output data')
    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
        format='%(levelname)s: %(asctime)s %(filename)s'
        ' [%(funcName)s:%(lineno)d][%(process)d] %(message)s',
        datefmt='%m-%d %H:%M:%S',
        filename=None,
        filemode='a')

    g_max_values = args.max_len[2]

    # loading dataSets
    lst_data, tables, dct_cells = utils.load_dataset(args.data_path, args.table_path, args.cell_path, args.max_len[0])

    g_parser = sql2grammar.Parser(args.grammar_path, args.max_len[0], args.max_len[1])

    # process datasets
    is_train = args.flag.lower() == 'true'
    process_result = preprocess(lst_data, dct_cells, g_parser, is_train)
    with open(args.output, 'w') as ofs:
        json.dump(lst_data, ofs, indent=2, ensure_ascii=False)


