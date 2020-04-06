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

"""trans sql to grammar rule sequence
"""

import sys
import os
import traceback
import logging
import argparse
import json
import jieba

import copy
import utils
from utils import CONST_COLUMN_DICT
import grammar


class Parser:
    """ Parser"""
    def __init__(self, grammar_file, max_table, max_column):
        self.copy_selec = None
        self.sel_result = []
        self.col_set = set()
        self.gmr2id = self._load_grammar2id(grammar_file)
        self.max_table = max_table
        self.max_column = max_column

    def parse(self, instance):
        """ parse the whole sql query
        Args:
            instance (dict): sql query in json format and db info.
                 for detail description of input instance, plearse refer to data/DuSQL_Desc.md
        """
        self._preprocess(instance)
        sql = instance['sql']
        query_toks_no_value = instance['query_toks_no_value']

        lst_single_sql = []
        lst_result = []

        if sql['intersect'] is not None:
            lst_result.append(grammar.SQL(0))
            lst_single_sql = [sql, sql['intersect']]
        elif sql['union'] is not None:
            lst_result.append(grammar.SQL(1))
            lst_single_sql = [sql, sql['union']]
        elif sql['except'] is not None:
            lst_result.append(grammar.SQL(2))
            lst_single_sql = [sql, sql['except']]
        elif self._is_row_calc(sql):
            op = sql['select'][0][1][0]
            sql_left = sql['from']['table_units'][0][1]
            sql_right = sql['from']['table_units'][1][1]

            if op == 1:         # -
                lst_result.append(grammar.SQL(4))
            elif op == 2:       # +
                lst_result.append(grammar.SQL(3))
            elif op == 3:       # *
                lst_result.append(grammar.SQL(5))
            elif op == 4:       # /
                lst_result.append(grammar.SQL(6))
            else:
                raise ValueError('unsupported row calc op: %d' % (op))
            lst_single_sql = [sql_left, sql_right]
        else:
            lst_result.append(grammar.SQL(7))
            lst_single_sql = [sql]

        for idx, single_sql in enumerate(lst_single_sql):
            sql_query = {}
            sql_query['sql'] = single_sql
            sql_query['column_names'] = instance['column_names']
            if idx > 0:
                query_toks_no_value = []
            sql_query['query_toks_no_value'] = query_toks_no_value
            sql_query['column_table'] = instance['column_table']
            sql_query['table_names'] = instance['table_names']
            sql_query['question'] = instance['question']
            sql_query['column_set'] = instance['column_set']
            sql_query['values'] = instance['values']
            sql_query['dct_values'] = instance['dct_values']
            lst_result.extend(self._parse_single_sql(sql_query))

        self._post_clean(instance)
        return lst_result

    def _preprocess(self, instance):
        """preprocess instance

        Args:
            instance (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        values = instance['values']
        dct_values = {}
        for v in values:
            v = ''.join(v)
            if self._is_number(v):
                v = float(v)
            dct_values[v] = len(dct_values)
        instance['dct_values'] = dct_values

    def _post_clean(self, instance):
        """clean keys

        Args:
            instance (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        del instance['dct_values']

    def grammar2id(self, rules):
        """ grammar to id
        Args:
            rules : label str
        """
        ruleid = []
        for rule in rules:
            rule = str(rule)
            if rule in self.gmr2id:
                ruleid.append(self.gmr2id[rule])
            elif rule.startswith('Table('):
                curr_idx = int(rule[:-1].split('(')[1])
                base = len(self.gmr2id)
                ruleid.append(str(base + curr_idx))
            elif rule.startswith('Column('):
                curr_idx = int(rule[:-1].split('(')[1])
                base = len(self.gmr2id) + self.max_table
                ruleid.append(str(base + curr_idx))
            elif rule.startswith('Value('):
                curr_idx = int(rule[:-1].split('(')[1])
                base = len(self.gmr2id) + self.max_table + self.max_column
                ruleid.append(str(base + curr_idx))
            else:
                raise ValueError('wrong grammar: %s from %s' % (rule, str(rules)))
        return ruleid

    def _parse_single_sql(self, query):
        """parse_single_sql
        Args:
            query:question json
        """
        stack = ["SingleSQL"]
        lst_result = []
        while len(stack) > 0:
            state = stack.pop()
            step_result, step_state = self._parse_step(state, query)
            lst_result.extend(step_result)
            if step_state:
                stack.extend(step_state)
        return lst_result

    def _parse_step(self, state, sql):
        """run one parse step"""
        if state == 'SingleSQL':
            return self._parse_root(sql)
        elif state == 'SELECT':
            return self._parse_select(sql)
        elif state == 'SUPERLATIVE':
            return self._parse_sup(sql)
        elif state == 'FILTER':
            return self._parse_filter(sql)
        elif state == 'ORDER':
            return self._parse_order(sql)
        else:
            raise NotImplementedError("Not the right state")

    def _parse_root(self, sql):
        """parsing the sql by the grammar
        R ::= Select | Select Filter | Select Order | ... |
        :return: [R(), states]
        """
        use_sup, use_ord, use_fil = True, True, False

        if sql['sql']['limit'] is None:
            use_sup = False

        if sql['sql']['orderBy'] == []:
            use_ord = False
        elif sql['sql']['limit'] is not None:
            use_ord = False

        # check the where and having
        if len(sql['sql']['where']) > 0 or len(sql['sql']['having']) > 0:
            use_fil = True

        if use_fil and use_sup:
            return [grammar.SingleSQL(0)], ['FILTER', 'SUPERLATIVE', 'SELECT']
        elif use_fil and use_ord:
            return [grammar.SingleSQL(1)], ['ORDER', 'FILTER', 'SELECT']
        elif use_sup:
            return [grammar.SingleSQL(2)], ['SUPERLATIVE', 'SELECT']
        elif use_fil:
            return [grammar.SingleSQL(3)], ['FILTER', 'SELECT']
        elif use_ord:
            return [grammar.SingleSQL(4)], ['ORDER', 'SELECT']
        else:
            return [grammar.SingleSQL(5)], ['SELECT']

    def _parse_select(self, sql):
        """
        parsing the sql by the grammar
        Select ::= Agg | AggAgg | AggAggAgg | ... |
        Agg ::= agg column table
        :return: [Select(), states]
        """
        lst_result = []
        select = sql['sql']['select']
        lst_result.append(grammar.Select(0))
        lst_result.append(grammar.NumA(len(select) - 1))

        for sel in select:
            if sel[1][0] == 0:
                lst_result.append(grammar.Agg(sel[0]))
                col_id, is_const_col = self._trans_col_id(sql['column_set'], sql['column_names'], sel[1][1][1])
                self.col_set.add(col_id)
                lst_result.append(grammar.Column(col_id))
                # now check for the situation with *
                if is_const_col:
                    lst_result.append(self._parse_const_column(sql, select))
                else:
                    lst_result.append(grammar.Table(sql['column_table'][sel[1][1][1]]))
            else:            #列计算
                lst_result.append(grammar.Agg(sel[0] + 6))
                if sel[1][0] == 1:      # -
                    lst_result.append(grammar.MathAgg(1))
                elif sel[1][0] == 2:    # +
                    lst_result.append(grammar.MathAgg(0))
                elif sel[1][0] == 3:    # *
                    lst_result.append(grammar.MathAgg(2))
                elif sel[1][0] == 4:    # /
                    lst_result.append(grammar.MathAgg(3))
                else:
                    raise NotImplementedError("Not the right MathAgg")
                lst_result.append(grammar.Agg(sel[1][1][0]))
                col_id1, is_const_col = self._trans_col_id(sql['column_set'], sql['column_names'], sel[1][1][1])
                lst_result.append(grammar.Column(col_id1))
                 # now check for the situation with *
                if is_const_col:
                    lst_result.append(self._parse_const_column(sql, select))
                else:
                    lst_result.append(grammar.Table(sql['column_table'][sel[1][1][1]]))
                lst_result.append(grammar.Agg(sel[1][2][0]))
                col_id2, is_const_col = self._trans_col_id(sql['column_set'], sql['column_names'], sel[1][2][1])
                lst_result.append(grammar.Column(col_id2))
                 # now check for the situation with *
                if is_const_col:
                    lst_result.append(self._parse_const_column(sql, select))
                else:
                    lst_result.append(grammar.Table(sql['column_table'][sel[1][1][1]]))
            if not self.copy_selec:
                self.copy_selec = [copy.deepcopy(lst_result[-2]), copy.deepcopy(lst_result[-1])]

        return lst_result, None

    def _parse_sup(self, sql):
        """
        parsing the sql by the grammar
        Superlative ::= Most Agg | Least Agg
        Agg ::= agg column table
        :return: [Superlative(), states]
        """
        lst_result = []
        if sql['sql']['limit'] is None:
            return lst_result, None
        if sql['sql']['orderBy'][0] == 'desc':
            lst_result.append(grammar.Superlative(0))
        else:
            lst_result.append(grammar.Superlative(1))

        agg_id, val_unit = sql['sql']['orderBy'][1][0]
        if val_unit[2] is not None:
            lst_result.append(grammar.Agg(agg_id + 6))

            if val_unit[0] == 1:
                lst_result.append(grammar.MathAgg(1))
            elif val_unit[0] == 2:
                lst_result.append(grammar.MathAgg(0))
            elif val_unit[0] == 3:
                lst_result.append(grammar.MathAgg(2))
            elif val_unit[0] == 4:
                lst_result.append(grammar.MathAgg(3))

            lst_result.extend(self._gen_agg(val_unit[1][0], val_unit[1], sql))
            lst_result.extend(self._gen_agg(val_unit[2][0], val_unit[2], sql))
        else:
            lst_result.extend(self._gen_agg(agg_id, val_unit[1], sql))

        value_idx = self._find_value_index(sql['dct_values'], sql['sql']['limit'])
        lst_result.append(grammar.Value(value_idx))
        return lst_result, None

    def _parse_filter(self, sql):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        Agg ::= agg column table
        :return: [Filter(), states]
        """
        lst_result = []

        cond_where = sql['sql']['where']
        cond_having = sql['sql']['having']
        if len(sql['sql']['where']) > 0 and len(sql['sql']['having']) > 0:
            lst_result.append(grammar.Filter(0))

        lst_result.extend(self._parse_conditions(sql, sql['sql']['where']))
        lst_result.extend(self._parse_conditions(sql, sql['sql']['having']))
        return lst_result, None

    def _parse_order(self, sql):
        """
        parsing the sql by the grammar
        Order ::= asc Agg | desc Agg
        Agg ::= agg column table
        :return: [Order(), states]
        """
        lst_result = []

        if 'order' not in sql['query_toks_no_value'] or 'by' not in sql['query_toks_no_value']:
            return lst_result, None
        elif 'limit' in sql['query_toks_no_value']:
            return lst_result, None

        if sql['sql']['orderBy'] == []:
            return lst_result, None

        if sql['sql']['orderBy'][0] == 'desc':
            lst_result.append(grammar.Order(0))
        else:
            lst_result.append(grammar.Order(1))

        agg_id, val_unit = sql['sql']['orderBy'][1][0]
        if val_unit[2] is not None:
            lst_result.append(grammar.Agg(agg_id + 6))

            if val_unit[0] == 1:
                lst_result.append(grammar.MathAgg(1))
            elif val_unit[0] == 2:
                lst_result.append(grammar.MathAgg(0))
            elif val_unit[0] == 3:
                lst_result.append(grammar.MathAgg(2))
            elif val_unit[0] == 4:
                lst_result.append(grammar.MathAgg(3))

            lst_result.extend(self._gen_agg(val_unit[1][0], val_unit[1], sql))
            lst_result.extend(self._gen_agg(val_unit[2][0], val_unit[2], sql))
        else:
            lst_result.extend(self._gen_agg(agg_id, val_unit[1], sql))

        return lst_result, None

    def _parse_const_column(self, sql, select):
        """
        Find table of column '*'
        :return: Table(table_id)
        """
        if len(sql['sql']['from']['table_units']) == 1:
            return grammar.Table(sql['sql']['from']['table_units'][0][1])

        table_list = []
        for tu in sql['sql']['from']['table_units']:
            if type(tu[1]) is int:
                table_list.append(tu[1])
        table_set, other_set = set(table_list), set()
        col2table = sql['column_table']
        for sel_p in select:
            if sel_p[1][1][1] != 0:
                other_set.add(col2table[sel_p[1][1][1]])

        for idx, condition in enumerate(sql['sql']['where']):
            if idx % 2 == 1:
                continue
            col_id = condition[2][1][1]
            other_set.add(col2table[col_id])
        #if len(sql['sql']['where']) == 1:
        #    other_set.add(col2table[sql['sql']['where'][0][2][1][1]])
        #elif len(sql['sql']['where']) == 3:
        #    other_set.add(col2table[sql['sql']['where'][0][2][1][1]])
        #    other_set.add(col2table[sql['sql']['where'][2][2][1][1]])
        #elif len(sql['sql']['where']) == 5:
        #    other_set.add(col2table[sql['sql']['where'][0][2][1][1]])
        #    other_set.add(col2table[sql['sql']['where'][2][2][1][1]])
        #    other_set.add(col2table[sql['sql']['where'][4][2][1][1]])
        table_set = table_set - other_set
        if len(table_set) == 1:
            return grammar.Table(list(table_set)[0])
        elif len(table_set) == 0 and sql['sql']['groupBy'] != []:
            return grammar.Table(col2table[sql['sql']['groupBy'][0][1]])
        else:
            question = sql['question']
            self.sel_result.append(question)
            logging.warning('find table of column * failed')
            return grammar.Table(sql['sql']['from']['table_units'][0][1])

    def _parse_conditions(self, sql, conditions):
        """parse filter condition list

        Args:
            sql (dict): NULL
            conditions (list): NULL

        Returns: int
                 condition number

        Raises: NULL
        """

        lst_result = []
        if len(conditions) == 0:
            return lst_result

        if len(conditions) == 1:
            lst_result.extend(self._parse_one_condition(conditions[0], sql['column_names'], sql))
        elif len(conditions) == 3:
            if conditions[1] == 'and':
                lst_result.append(grammar.Filter(0))
            else:
                lst_result.append(grammar.Filter(1))
            lst_result.extend(self._parse_one_condition(conditions[0], sql['column_names'], sql))
            lst_result.extend(self._parse_one_condition(conditions[2], sql['column_names'], sql))
        elif len(conditions) == 5:
            if conditions[1] == 'and' and conditions[3] == 'and':
                lst_result.append(grammar.Filter(0))
                lst_result.extend(self._parse_one_condition(conditions[0], sql['column_names'], sql))
                lst_result.append(grammar.Filter(0))
                lst_result.extend(self._parse_one_condition(conditions[2], sql['column_names'], sql))
                lst_result.extend(self._parse_one_condition(conditions[4], sql['column_names'], sql))
            elif conditions[1] == 'and' and conditions[3] == 'or':
                lst_result.append(grammar.Filter(1))
                lst_result.append(grammar.Filter(0))
                lst_result.extend(self._parse_one_condition(conditions[0], sql['column_names'], sql))
                lst_result.extend(self._parse_one_condition(conditions[2], sql['column_names'], sql))
                lst_result.extend(self._parse_one_condition(conditions[4], sql['column_names'], sql))
            elif conditions[1] == 'or' and conditions[3] == 'and':
                lst_result.append(grammar.Filter(1))
                lst_result.append(grammar.Filter(0))
                lst_result.extend(self._parse_one_condition(conditions[2], sql['names'], sql))
                lst_result.extend(self._parse_one_condition(conditions[4], sql['column_names'], sql))
                lst_result.extend(self._parse_one_condition(conditions[0], sql['column_names'], sql))
            else:
                lst_result.append(grammar.Filter(1))
                lst_result.append(grammar.Filter(1))
                lst_result.extend(self._parse_one_condition(conditions[0], sql['column_names'], sql))
                lst_result.extend(self._parse_one_condition(conditions[2], sql['column_names'], sql))
                lst_result.extend(self._parse_one_condition(conditions[4], sql['column_names'], sql))
        elif len(conditions) % 2 == 0:
            raise ValueError('conditions\'s length must be odd. but got %d for %s' % (len(conditions), str(conditions)))
        return lst_result

    def _parse_one_condition(self, cond_unit, names, sql):
        """ parse one condition unit
        Args:
            cond_unit: cond_unit
            names:names
            sql:sql
        """
        lst_result = []
        # check if V(root)
        nest_query = True
        if type(cond_unit[3]) != dict:
            nest_query = False

        # check for Filter (=, <, >, !=, between, >=, <=, ...)
        # 00: not_in
        # 01: between   05: >=    09: like
        # 02: =         06: <=    10: is
        # 03: >         07: !=    11: exists
        # 04: <         08: in    12: not like
        # map from sql op id to grammar filter id
        single_map = {1: 8, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 3}
        nested_map = {1: 15, 2: 11, 3: 13, 4: 12, 5: 16, 6: 17, 7: 14}
        cond_op = cond_unit[1]
        if cond_op in [1, 2, 3, 4, 5, 6, 7]:
            if nest_query == False:
                fil = grammar.Filter(single_map[cond_op])
            else:
                fil = grammar.Filter(nested_map[cond_op])
        elif cond_op == 8:
            fil = grammar.Filter(18)
        elif cond_op == 9:
            fil = grammar.Filter(9)
        elif cond_op == 12:
            fil = grammar.Filter(10)
        elif cond_op == 0:
            fil = grammar.Filter(19)
        else:
            raise NotImplementedError("not implement for the others FIL")

        lst_result.append(fil)
        # MathAgg
        if cond_unit[2][2] is not None:
            lst_result.append(grammar.Agg(cond_unit[0] + 6))

            if cond_unit[2][0] == 1:
                lst_result.append(grammar.MathAgg(1))
            elif cond_unit[2][0] == 2:
                lst_result.append(grammar.MathAgg(0))
            elif cond_unit[2][0] == 3:
                lst_result.append(grammar.MathAgg(2))
            elif cond_unit[2][0] == 4:
                lst_result.append(grammar.MathAgg(3))

            lst_result.extend(self._gen_agg(cond_unit[2][1][0], cond_unit[2][1], sql))
            lst_result.extend(self._gen_agg(cond_unit[2][2][0], cond_unit[2][2], sql))
        else:
            lst_result.extend(self._gen_agg(cond_unit[0], cond_unit[2][1], sql))

        dct_values = sql['dct_values']
        if not nest_query:
            lst_result.append(grammar.Value(self._find_value_index(dct_values, cond_unit[3])))
        if not nest_query and cond_unit[4] is not None:
            lst_result.append(grammar.Value(self._find_value_index(dct_values, cond_unit[4])))

        if nest_query:
            nest_query = {}
            nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = cond_unit[3]
            nest_query['column_table'] = sql['column_table']
            nest_query['column_names'] = sql['column_names']
            nest_query['table_names'] = sql['table_names']
            nest_query['question'] = sql['question']
            nest_query['column_set'] = sql['column_set']
            nest_query['values'] = sql['values']
            nest_query['dct_values'] = sql['dct_values']
            lst_result.extend(self._parse_single_sql(nest_query))

        return lst_result

    def _find_value_index(self, dct_value2index, val):
        """find value index from dct_value2index

        Args:
            dct_value2index (TYPE): NULL
            val (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        if self._is_number(val):
            val = float(val)
        if val in dct_value2index:
            return dct_value2index[val]

        max_score = -1
        max_idx = -1
        ref_val_set = set(str(val))
        for candi_val, idx in dct_value2index.items():
            if type(candi_val) is float:
                continue
            score = len(set(candi_val) & ref_val_set) / len(candi_val)
            if score > max_score:
                max_score = score
                max_idx = idx
        return max_idx

    def _gen_agg(self, agg_id, col_unit, sql):
        """gen agg grammar
        Args:
            agg_id (TYPE): NULL
            col_unit (tuple): (agg_id, col_id), agg_id here is useless.
            sql (TYPE): NULL
        Returns: TODO
        """
        result = []
        result.append(grammar.Agg(agg_id))
        col_id, is_const_col = self._trans_col_id(sql['column_set'], sql['column_names'], col_unit[1])
        self.col_set.add(col_id)
        result.append(grammar.Column(col_id))
        if is_const_col:
            select = sql['sql']['select']
            result.append(self._parse_const_column(sql, select))
        else:
            result.append(grammar.Table(sql['column_table'][col_unit[1]]))
        return result

    def _trans_col_id(self, column_set, column_list, origin_col_id):
        """trans column id in column_list to column_set

        Args:
            column_set (list): column names list removed duplicated
            column_list (list): original column names list
            origin_col_id (int/str): index of column in column_list, or const names, like TIME_NOW

        Returns: tuple (new_column_id, is_const_column)
                 const column: */TIME_NOW

        Raises: NULL
        """
        if type(origin_col_id) is int:
            origin_column_name = ''.join(column_list[origin_col_id])
            return column_set.index(origin_column_name), origin_col_id == 0

        if origin_col_id not in CONST_COLUMN_DICT:
            raise ValueError('illegal const column: %s' % (origin_col_id))

        col_name = CONST_COLUMN_DICT[origin_col_id]
        if column_set[-1] != col_name:
            column_set.append(col_name)
        return len(column_set) - 1, True

    def _load_grammar2id(self, grammar_file):
        """load grammar2id

        Args:
            grammar_file (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        with open(grammar_file) as ifs:
            lst_grammar = [x.strip() for x in ifs if not x.startswith('#')]

        curr_id = -1
        curr_tag = ''
        dct_gmr2id = dict()
        for rule in lst_grammar:
            name = rule.split(' ', 1)[0]
            if name in ('Table', 'Column', 'Value'):
                continue
            if name != curr_tag:
                curr_tag = name
                curr_id = 0
            key = '%s(%d)' % (name, curr_id)
            logging.info('adding rule id %2d <-- %s (%s)', len(dct_gmr2id), key, rule)
            dct_gmr2id[key] = len(dct_gmr2id)
            curr_id += 1
        return dct_gmr2id

    def _is_row_calc(self, sql):
        """check whether sql is performing row calculating

        Args:
            sql (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        table_units = sql['from']['table_units']
        ## 行计算的 SQL from 部分包含 2 个子 SQL
        if not (len(table_units) == 2 and table_units[0][0] == 'sql' and table_units[1][0] == 'sql'):
            return False

        select = sql['select']
        # select[0]: (agg_id, val_unit)
        #            val_unit: (calc_op, col_unit1, col_unit2)
        if not (len(select) == 1 and select[0][1][0] > 0 and select[0][1][1] == select[0][1][2]):
            return False
        return True

    def _is_number(self, n):
        """is number

        Args:
            n (TYPE): NULL

        Returns: bool

        Raises: NULL
        """
        try:
            n = float(n)
            return True
        except:
            return False


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--cell_path', type=str, help='cell dataset', required=True)
    arg_parser.add_argument('--grammar-path', required=True, help='grammar file path')
    arg_parser.add_argument('--max-len', required=True, nargs=2, type=int,
                                               help='max len of table and column. like "-g 20 60"')
    arg_parser.add_argument('--output', type=str, help='output data', required=True)
    args = arg_parser.parse_args()

    parser = Parser(args.grammar_path, args.max_len[0], args.max_len[1])

    # loading dataSets
    datas, table, cells= utils.load_dataset(args)
    processed_data = []

    for i, d in enumerate(datas):
        if len(datas[i]['sql']['select']) > 5:
            continue
        r = parser.parse(datas[i])
        r = parser.grammar2id(r)
        datas[i]['rule_label'] = " ".join([str(x) for x in r])
        processed_data.append(datas[i])

    print('Finished %s datas and failed %s datas' % (len(processed_data), len(datas) - len(processed_data)))
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(json.dumps(processed_data, ensure_ascii=False))

