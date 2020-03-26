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

"""trans grammar rule id to sql query
"""

import argparse
import traceback
import re
import json
import logging
import ast

import graph 
import utils
import grammar_utils
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

DFT_RULE = 'SQL(7) SingleSQL(5) Select(0) NumA(0) Agg(3) Column(0) Table(0)'
CONST_COLUMN_DICT = dict([(v, k) for k, v in utils.CONST_COLUMN_DICT.items()])

g_direction_map = {"des": 'DESC', 'asc': 'ASC'}
g_agg_tag_list = ['count(', 'avg(', 'min(', 'max(', 'sum(']

def pop_front(array):
    """ pop
    """
    if len(array) == 0:
        return 'None'
    return array.pop(0)


def is_end(components, transformed_sql, is_root_processed):
    """ judge is end of grammar or not
    Args:
        components(list): logical form list
        transformed_sql(dict): 
        is_root_processed(bool):True/False
    Returns: True/False
    Rsises: NULL
    """
    end = False
    c = pop_front(components)
    c_instance = eval(c)
    if isinstance(c_instance, SingleSQL) and is_root_processed:
        # intersect, union, except
        end = True
    elif isinstance(c_instance, Filter):
        if 'where' not in transformed_sql:
            end = True
        else:
            num_conjunction = 0
            for f in transformed_sql['where']:
                if isinstance(f, str) and (f == 'and' or f == 'or'):
                    num_conjunction += 1
            current_filters = len(transformed_sql['where'])
            valid_filters = current_filters - num_conjunction
            if valid_filters >= num_conjunction + 1:
                end = True
    elif isinstance(c_instance, Order):
        if 'order' not in transformed_sql:
            end = True
        elif len(transformed_sql['order']) == 0:
            end = False
        else:
            end = True
    elif isinstance(c_instance, Superlative):
        if 'sup' not in transformed_sql:
            end = True
        elif len(transformed_sql['sup']) == 0:
            end = False
        else:
            end = True
    components.insert(0, c)
    return end


def _rename_const_columns(column_names):
    """rename meaningful const column name to origin

    Args:
        column_names (list): NULL

    Returns: TODO

    Raises: NULL
    """
    for idx, name_toks in enumerate(column_names):
        name = ''.join(name_toks)
        if name in CONST_COLUMN_DICT:
            column_names[idx] = [CONST_COLUMN_DICT[name]]


def _transform_num_a(c_instance, components, transformed_sql, col_set, table_names, values, schema, current_table):
    """deal _transform NumA"""
    for i in range(c_instance.id_c + 1):
        agg = eval(pop_front(components))
        assert isinstance(agg, Agg)
        temp  = eval(pop_front(components))
        if isinstance(temp, Column):
            column = temp
            _table = pop_front(components)
            table = eval(_table)
            if not isinstance(table, Table):
                table = None
                components.insert(0, _table)
            # agg_id, mathagg_id, agg_id, column, table
            col_name = col_set[column.id_c]
            if table is not None:
                table_name = table_names[table.id_c]
                origin_col = replace_col_with_original_col(col_name, table_name, current_table)
            else:
                table_name = None
                origin_col = col_name
            transformed_sql['select'].append((
                -1, -1,
                agg.production.split()[1],
                origin_col,
                table_name
            ))
        elif isinstance(temp, MathAgg):
            magg = agg
            for _ in range(2):
                agg = eval(pop_front(components))
                column = eval(pop_front(components))
                _table = pop_front(components)
                table = eval(_table)
                if not isinstance(table, Table):
                    table = None
                    components.insert(0, _table)
                assert isinstance(agg, Agg) and isinstance(column, Column)
                transformed_sql['select'].append((
                    magg.production.split()[1],
                    temp.production.split()[1],
                    agg.production.split()[1],
                    replace_col_with_original_col(col_set[column.id_c],
                    table_names[table.id_c], current_table) if table is not None else col_set[column.id_c], 
                    table_names[table.id_c] if table is not None else table
                ))
 
def _transform_sup(c_instance, components, transformed_sql, col_set, table_names, values, schema, current_table):
    """deal _transform superlative"""
    transformed_sql['sup'].append(c_instance.production.split()[1])
    agg = eval(pop_front(components))
    assert isinstance(agg, Agg)
    temp = eval(pop_front(components))
    if isinstance(temp, Column):
        column = temp
        table = eval(pop_front(components))
        assert isinstance(table, Table)
        transformed_sql['sup'].append(agg.production.split()[1])
        fix_col_id = replace_col_with_original_col(col_set[column.id_c], 
                table_names[table.id_c], current_table)
        transformed_sql['sup'].append(fix_col_id)
        transformed_sql['sup'].append(table_names[table.id_c])
    elif isinstance(temp, MathAgg):
        magg = agg.production.split()[1]
        mathagg = temp.production.split()[1]
        transformed_sql['sup'].append(magg)
        transformed_sql['sup'].append(mathagg)
        for _ in range(2):
            agg = eval(pop_front(components))
            column = eval(pop_front(components))
            table = eval(pop_front(components))
            assert isinstance(agg, Agg) and isinstance(column, Column) and isinstance(table, Table)
            transformed_sql['sup'].append(agg.production.split()[1])
            fix_col_id = replace_col_with_original_col(col_set[column.id_c], 
                    table_names[table.id_c], current_table)
            transformed_sql['sup'].append(fix_col_id)
            transformed_sql['sup'].append(table_names[table.id_c] if table is not None else table)
    value = eval(pop_front(components))
    assert isinstance(value, Value)
    transformed_sql['sup'].append(''.join(values[value.id_c]))


def  _transform_order(c_instance, components, transformed_sql, col_set, table_names, values, schema, current_table):
    """ deal _transform order"""
    transformed_sql['order'].append(c_instance.production.split()[1])
    agg = eval(pop_front(components))
    assert isinstance(agg, Agg)
    temp = eval(pop_front(components))
    if isinstance(temp, Column):
        column = temp
        _table = pop_front(components)
        table = eval(_table)
        if not isinstance(table, Table):
            table = None
            components.insert(0, _table)
        transformed_sql['order'].append(agg.production.split()[1])
        transformed_sql['order'].append(replace_col_with_original_col(col_set[column.id_c], 
                                        table_names[table.id_c], current_table))
        transformed_sql['order'].append(table_names[table.id_c] if table is not None else table)
    elif isinstance(temp, MathAgg):
        magg = agg.production.split()[1]
        mathagg = temp.production.split()[1]
        transformed_sql['order'].append(magg)
        transformed_sql['order'].append(mathagg)
        for _ in range(2):
            agg = eval(pop_front(components))
            column = eval(pop_front(components))
            _table = pop_front(components)
            table = eval(_table)
            if not isinstance(table, Table):
                table = None
                components.insert(0, _table)
            assert isinstance(agg, Agg) and isinstance(column, Column)
            transformed_sql['order'].append(agg.production.split()[1])
            transformed_sql['order'].append(replace_col_with_original_col(col_set[column.id_c], 
                table_names[table.id_c], current_table))
            transformed_sql['order'].append(table_names[table.id_c] if table is not None else table)

 
def _transform_filter(c_instance, components, transformed_sql, col_set, table_names, values, schema, current_table):
    """deal _transform function's Filter"""
    op = c_instance.production.split()[1]
    ## in case of:
    #    Filter --> and Filter Filter
    #    Filter --> or Filter Filter
    if op == 'and' or op == 'or':
        transformed_sql['where'].append(op)
        return

    # No Supquery
    agg = eval(pop_front(components))
    assert isinstance(agg, Agg)
    # Column or MathAgg
    temp = eval(pop_front(components))
    if isinstance(temp, Column):
        column = temp
        _table = pop_front(components)
        table = eval(_table)
        if not isinstance(table, Table):
            table = None
            components.insert(0, _table)

        root_agg_t, math_agg_t = -1, -1
        sub_agg_t = agg.production.split()[1]
        table_name = table_names[table.id_c]
        col_id = replace_col_with_original_col(col_set[column.id_c], table_name, current_table)
    elif isinstance(temp, MathAgg):
        agg1 = eval(pop_front(components))
        col1 = eval(pop_front(components))
        table1 = eval(pop_front(components))
        assert isinstance(agg1, Agg) and isinstance(col1, Column) and isinstance(table1, Table)
        agg2 = eval(pop_front(components))
        col2 = eval(pop_front(components))
        table2 = eval(pop_front(components))
        assert isinstance(agg2, Agg) and isinstance(col2, Column) and isinstance(table2, Table)

        root_agg_t = agg.production.split()[1]
        math_agg_t = temp.production.split()[1]
        sub_agg_t = (agg1.production.split()[1], agg2.production.split()[1])
        table_name = (table_names[table1.id_c], table_names[table2.id_c])
        col_id = (replace_col_with_original_col(col_set[col1.id_c], table_name[0], current_table),
                  replace_col_with_original_col(col_set[col2.id_c], table_name[1], current_table))

    if c_instance.production.split()[3] == 'Value':
        val_rule = pop_front(components)
        value_id = grammar_utils._id(val_rule)
        value = values[value_id]

        # magg, mathagg, op, agg, column, table, value, column_start
        transformed_sql['where'].append((
            root_agg_t, math_agg_t,
            op,
            sub_agg_t,
            col_id,
            table_name,
            value
        ))
        #### note that, 'between' grammar is not processed here!
    else:           # nested sql
        new_dict = dict()
        new_dict['sql'] = transformed_sql['sql']
        transformed_sql['where'].append((
            root_agg_t, math_agg_t,
            op,
            sub_agg_t,
            col_id,
            table_name,
            _transform(components, new_dict, col_set, table_names, values, schema)
        ))
 

def _transform(components, transformed_sql, col_set, table_names, values, schema):
    """ transform grammar to sql 
    Args:
        components(list): logical form list
        transformed_sql(dict): 
        col_set(list): columns_names set
        table_names(list): table_names
        values(list): values
        schema(dict): schema
    Returns: transformed_sql
    Rsises: NULL
    """
    processed_root = False
    current_table = schema
    while len(components) > 0:
        if is_end(components, transformed_sql, processed_root):
            break
        c = pop_front(components)
        c_instance = eval(c)
        if isinstance(c_instance, SingleSQL):
            processed_root = True
            transformed_sql['select'] = list()
            if c_instance.id_c == 0:
                transformed_sql['where'] = list()
                transformed_sql['sup'] = list()
            elif c_instance.id_c == 1:
                transformed_sql['where'] = list()
                transformed_sql['order'] = list()
            elif c_instance.id_c == 2:
                transformed_sql['sup'] = list()
            elif c_instance.id_c == 3:
                transformed_sql['where'] = list()
            elif c_instance.id_c == 4:
                transformed_sql['order'] = list()
        elif isinstance(c_instance, Select):
            continue
        elif isinstance(c_instance, NumA):
            _transform_num_a(
                    c_instance, components, transformed_sql, col_set, table_names, values, schema, current_table)
        elif isinstance(c_instance, Superlative):
            _transform_sup(
                    c_instance, components, transformed_sql, col_set, table_names, values, schema, current_table)
        elif isinstance(c_instance, Order):
            _transform_order(
                    c_instance, components, transformed_sql, col_set, table_names, values, schema, current_table)
        elif isinstance(c_instance, Filter):
            _transform_filter(
                    c_instance, components, transformed_sql, col_set, table_names, values, schema, current_table)
        else:
            raise TypeError('unsupported grammar rule instance: %s' % (c))
    return transformed_sql


def transform(query, schema, origin=None):
    """ transform
    Args:
        query(dict): query
        schema(dict): schema
        origin(list): logical form
    Returns: parse result
    Rsises: NULL
    """
    preprocess_schema(schema)
    if origin is None:
        lf = query['predicted_rule_tokens']
    else:
        lf = origin
    #_rename_const_columns(query['column_names'])
    col_set = query['column_names']
    table_names = query['table_names']
    values = query['values']
    current_table = schema

    current_table['schema_content_clean'] = [x[1] for x in current_table['column_names']]
    current_table['schema_content'] = [x[1] for x in current_table['column_names']]

    components = lf.split()
    transformed_sql = dict()
    transformed_sql['sql'] = query
    c = pop_front(components)
    c_instance = eval(c)
    assert isinstance(c_instance, SQL)
    if c_instance.id_c == 0:        # intersect
        transformed_sql['intersect'] = dict()
        transformed_sql['intersect']['sql'] = query
        _transform(components, transformed_sql, col_set, table_names, values, schema)
        _transform(components, transformed_sql['intersect'], col_set, table_names, values, schema)
    elif c_instance.id_c == 1:      # union
        transformed_sql['union'] = dict()
        transformed_sql['union']['sql'] = query
        _transform(components, transformed_sql, col_set, table_names, values, schema)
        _transform(components, transformed_sql['union'], col_set, table_names, values, schema)
    elif c_instance.id_c == 2:      # except
        transformed_sql['except'] = dict()
        transformed_sql['except']['sql'] = query
        _transform(components, transformed_sql, col_set, table_names, values, schema)
        _transform(components, transformed_sql['except'], col_set, table_names, values, schema)
    elif c_instance.id_c == 7:
        _transform(components, transformed_sql, col_set, table_names, values, schema)
    else:      # row calc +,-,*,/
        assert c_instance.id_c >= 3 and c_instance.id_c <= 6
        op = ['+', '-', '*', '/'][c_instance.id_c - 3]
        transformed_sql['row_calc_op'] = op
        transformed_sql['row_calc_left'] = dict()
        transformed_sql['row_calc_right'] = dict()
        transformed_sql['row_calc_left']['sql'] = query
        transformed_sql['row_calc_right']['sql'] = query
        _transform(components, transformed_sql['row_calc_left'], col_set, table_names, values, schema)
        _transform(components, transformed_sql['row_calc_right'], col_set, table_names, values, schema)

    parse_result = to_str_main(transformed_sql, 1, schema)

    parse_result = parse_result.replace('\t', '').strip()
    return [parse_result]


def col_to_str(agg, col, tab, table_names, N=1):
    """ transform Agg Column Table to str
    Args:
        agg(str): 
        col(str):
        tab(str):
        table_names(dict):
        N(int): Default is 1
    Returns: str
    Rsises: NULL
    """
    _col = col.replace(' ', '_')
    tab = ''.join(tab)
    if agg == 'none':
        if tab not in table_names:
            table_names[tab] = 'Table' + str(len(table_names) + N)
        table_alias = table_names[tab]
        if col == '*':
            return '*'
        return '%s.%s' % (table_alias, _col)

    if col == '*':
        if tab is not None and tab not in table_names:
            table_names[tab] = 'Table' + str(len(table_names) + N)
        return '%s(%s)' % (agg, _col)
    else:
        if tab not in table_names:
            table_names[tab] = 'Table' + str(len(table_names) + N)
        table_alias = table_names[tab]
        return '%s(%s.%s)' % (agg, table_alias, _col)


def infer_from_clause(table_names, schema, columns):
    """ infer from clause
    Args:
        table_names(list): 
        schema(dict):
        columns(list):
    Returns: str
    Rsises: NULL
    """
    tables = list(table_names.keys())
    start_table = None
    end_table = None
    join_clause = list()
    if len(tables) == 1:
        join_clause.append((tables[0], table_names[tables[0]]))
    elif len(tables) == 2:
        use_graph = True
        for t in tables:
            if t not in schema['graph'].vertices:
                use_graph = False
                break
        if use_graph:
            start_table = tables[0]
            end_table = tables[1]
            _tables = list(schema['graph'].dijkstra(tables[0], tables[1]))
            max_key = 1
            for t, k in table_names.items():
                _k = int(k[5:])
                if _k > max_key:
                    max_key = _k
            for t in _tables:
                if t not in table_names:
                    table_names[t] = 'Table' + str(max_key + 1)
                    max_key += 1
                join_clause.append((t, table_names[t],))
        else:
            join_clause = list()
            for t in tables:
                join_clause.append((t, table_names[t],))
    else:
        for t in tables:
            join_clause.append((t, table_names[t],))

    if len(join_clause) >= 3:
        star_table = None
        for agg, col, tab in columns:
            if col == '*':
                star_table = tab
                break
        if star_table is not None:
            star_table_count = 0
            for agg, col, tab in columns:
                if tab == star_table and col != '*':
                    star_table_count += 1
            if star_table_count == 0 and ((end_table is None or end_table == star_table) or \
                    (start_table is None or start_table == star_table)):
                # Remove the table the rest tables still can join without star_table
                new_join_clause = list()
                for t in join_clause:
                    if t[0] != star_table:
                        new_join_clause.append(t)
                join_clause = new_join_clause

    join_clause = ' JOIN '.join(['%s AS %s' % (jc[0], jc[1]) for jc in join_clause])
    return 'FROM ' + join_clause


def replace_col_with_original_col(col_name, table_name, current_table):
    """ replace col with original col
    Args:
        col_name(dict): 
        table_name(list):
        current_table(str):
    Returns: single_final_col
    Rsises: NULL
    """
    col_name = ''.join(col_name)
    if col_name == '*':
        return col_name
    if col_name in CONST_COLUMN_DICT:
        return CONST_COLUMN_DICT[col_name]

    cur_table = ''.join(table_name)
    new_col = None
    single_final_col = None
    for col_ind, cname in enumerate(current_table['schema_content_clean']):
        if cname != col_name:
            continue
        assert cur_table in current_table['table_names']
        col_table_id = current_table['col_table'][col_ind]
        new_col = current_table['column_names'][col_ind][1]
        if current_table['table_names'][col_table_id] == cur_table:
            single_final_col = new_col
            break

    if single_final_col is None:
        single_final_col = new_col
    if single_final_col is None:
        single_final_col = col_name
    return single_final_col


def build_graph(schema):
    """build graph
    Args:
        schema(dict):
    Returns: graph
    Rsises: NULL
    """
    relations = list()
    foreign_keys = schema['foreign_keys']
    for (fkey, pkey) in foreign_keys:
        fkey_table = schema['table_names'][schema['column_names'][fkey][0]]
        pkey_table = schema['table_names'][schema['column_names'][pkey][0]]
        relations.append((fkey_table, pkey_table))
        relations.append((pkey_table, fkey_table))
    return graph.Graph(relations)


def preprocess_schema(schema):
    """preprocess_schema
    Args:
        schema(dict):
    Returns: NULL
    Rsises: NULL
    """
    tmp_col = []
    for cc in [x[1] for x in schema['column_names']]:
        if cc not in tmp_col:
            tmp_col.append(cc)
    schema['column_set'] = tmp_col
    schema['schema_content'] = [col[1] for col in schema['column_names']]
    schema['col_table'] = [col[0] for col in schema['column_names']]
    graph = build_graph(schema)
    schema['graph'] = graph


def _select_to_str(select_list, N_T, table_names, all_columns):
    """structed select list to string

    Args:
        select_list (TYPE): NULL
        N_T (int): NULL
        table_names (dict): [in/out]
        all_columns (list): [in/out]

    Returns: TODO

    Raises: NULL
    """
    select_id = 0
    lst_select_item = []
    while select_id < len(select_list):
        (magg, mathagg, agg, col, tab) = select_list[select_id]
        all_columns.append((agg, col, tab))
        if magg == -1:
            select_id = select_id + 1
            lst_select_item.append(col_to_str(agg, col, tab, table_names, N_T))
        else:
            (magg1, mathagg1, agg1, col1, tab1) = select_list[select_id + 1]
            select_id = select_id + 2
            if magg1 == 'none':
                lst_select_item.append(col_to_str(agg, col, tab, table_names, N_T) + \
                                       ' ' + mathagg1 + ' ' + \
                                       col_to_str(agg1, col1, tab1, table_names, N_T))
            else:
                lst_select_item.append(magg1 + '(' + \
                                       col_to_str(agg, col, tab, table_names, N_T) + ' ' + \
                                       mathagg1 + ' ' + \
                                       col_to_str(agg1, col1, tab1, table_names, N_T) + ')')
    select_clause = 'SELECT ' + ', '.join(lst_select_item).strip()
    return select_clause, lst_select_item


def _order_to_str(order_list, N_T, table_names, all_columns, need_limit=False):
    """structed order/sup list to string

    Args:
        order_list (TYPE): NULL
        N_T (int): NULL
        table_names (dict): [in/out]
        all_columns (list): [in/out]
        need_limit (TYPE): Default is False

    Returns: TODO

    Raises: NULL
    """
    order_clause = ''
    limit_val = None
    if len(order_list) == 5:
        limit_val = order_list.pop(4)
    elif len(order_list) == 10:
        limit_val = order_list.pop(9)

    if len(order_list) == 4:
        (direction, agg, col, tab,) = order_list
        all_columns.append((agg, col, tab))
        subject = col_to_str(agg, col, tab, table_names, N_T)
        order_clause = ('ORDER BY %s %s' % (subject, g_direction_map[direction])).strip()
    elif len(order_list) == 9:
        (direction, magg, mathagg, agg1, col1, tab1, agg2, col2, tab2) = order_list
        all_columns.append((agg1, col1, tab1))
        all_columns.append((agg2, col2, tab2))
        subject1 = col_to_str(agg1, col1, tab1, table_names, N_T)
        subject2 = col_to_str(agg2, col2, tab2, table_names, N_T)
        if magg == 'none':
            order_clause = ('ORDER BY (%s %s %s) %s' % \
                    (subject1, mathagg, subject2, g_direction_map[direction])).strip()
        else:
            order_clause = ('ORDER BY %s(%s %s %s) %s' % \
                    (magg, subject1, mathagg, subject2, g_direction_map[direction])).strip()
    else:
        raise ValueError('order list length must be 4 or 9, got %d' % (len(order_list)))

    if limit_val is not None:
        order_clause += ' LIMIT %s' % (limit_val)
    return order_clause


def _condition_to_str(condition_list, N_T, root_sql, schema, table_names, all_columns):
    """structed condition(where and/or having) to string

    Args:
        condition_list (TYPE): NULL
        N_T (int): NULL
        root_sql (dict): root sql json dict
        schema (dict): 
        table_names (dict): [in/out]
        all_columns (list): [in/out]

    Returns: tuple (where clause, having clause)

    Raises: NULL
    """
    where_clause, having_clause = '', ''
    if len(condition_list) == 0:
        return where_clause, having_clause

    conjunctions = list()
    filters = list()
    for f in condition_list:
        if isinstance(f, str):
            conjunctions.append(f)
            continue

        root_agg, math_agg, op, agg, col, tab, value = f
        if op == '=':
            op = '=='
        if value and type(value) is dict:
            ## TODO: 需要检查这个逻辑
            value['sql'] = root_sql
        if value == 'None' or value is None or type(value) is list:
            if not type(value) is list:
                where_value = '1'
            else:
                where_value = ''.join(value)
            patt = re.compile('^[+-]?\d+(\.\d+)?%?$')
            if not patt.match(where_value):
                where_value = "'%s'" % (where_value)
            elif where_value.endswith('%'):
                where_value = str(float(where_value[:-1]) / 100)

            if root_agg == -1 and math_agg == -1:
                all_columns.append((agg, col, tab))
                subject = col_to_str(agg, col, tab, table_names, N_T)
                filters.append("%s %s %s" % (subject, op, where_value))
            else:
                agg1, agg2 = agg
                col1, col2 = col
                tab1, tab2 = tab
                all_columns.append((agg1, col1, tab1))
                all_columns.append((agg2, col2, tab2))
                subject1 = col_to_str(agg1, col1, tab1, table_names, N_T)
                subject2 = col_to_str(agg2, col2, tab2, table_names, N_T)
                filter_str = '(%s %s %s) %s %s' % (subject1, math_agg, subject2, op, where_value)
                if root_agg.lower() != 'none':
                    filter_str = root_agg + filter_str
                filters.append(filter_str)
        else:
            if op == 'in' and len(value['select']) == 1 and value['select'][0][0] == 'none' \
                    and 'where' not in value and 'order' not in value and 'sup' not in value:
                    # and value['select'][0][2] not in table_names:
                if value['select'][0][2] not in table_names:
                    table_names[value['select'][0][2]] = 'Table' + str(len(table_names) + N_T)
                filters.append(None)
            elif root_agg == -1 and math_agg == -1:
                all_columns.append((agg, col, tab))
                subject = col_to_str(agg, col, tab, table_names, N_T)
                filters.append("%s %s (%s)" % (subject, op, to_str(value, len(table_names) + 1, schema).strip()))
            else:
                agg1, agg2 = agg
                col1, col2 = col
                tab1, tab2 = tab
                all_columns.append((agg1, col1, tab1))
                all_columns.append((agg2, col2, tab2))
                subject1 = col_to_str(agg1, col1, tab1, table_names, N_T)
                subject2 = col_to_str(agg2, col2, tab2, table_names, N_T)
                filter_str='(%s %s %s) %s (%s)' % \
                           (subject1, math_agg, subject2, op, to_str(value, len(table_names) + 1, schema).strip())
                filters.append(filter_str)
        if len(conjunctions):
            filters.append(conjunctions.pop())

    aggs = ['count(', 'avg(', 'min(', 'max(', 'sum(']
    having_filters = list()
    idx = 0
    while idx < len(filters):
        _filter = filters[idx]
        if _filter is None:
            idx += 1
            continue
        for agg in aggs:
            if _filter.startswith(agg):
                having_filters.append(_filter)
                filters.pop(idx)
                if 0 < idx and (filters[idx - 1] in ['and', 'or']):
                    having_filters.insert(1, filters[idx - 1])
                    filters.pop(idx - 1)
                break
        else:
            idx += 1
    if len(having_filters) > 0:
        if len(having_filters) == 1:
            having_clause = 'HAVING ' + ' '.join(having_filters).strip()
        elif len(having_filters) == 3:
            having_clause = 'HAVING ' + having_filters[0].strip() + ' ' +  \
            having_filters[1].strip() + ' ' + having_filters[2].strip()
        elif len(having_filters) == 2:
            having_clause = 'HAVING ' + having_filters[0].strip() 
    if len(filters) > 0:
        filters = [_f for _f in filters if _f is not None]
        conjun_num = 0
        filter_num = 0
        for _f in filters:
            if _f in ['or', 'and']:
                conjun_num += 1
            else:
                filter_num += 1
        if conjun_num > 0 and filter_num != (conjun_num + 1):
            # assert 'and' in filters
            idx = 0
            while idx < len(filters):
                if filters[idx] == 'and':
                    if idx - 1 == 0:
                        filters.pop(idx)
                        break
                    if filters[idx - 1] in ['and', 'or']:
                        filters.pop(idx)
                        break
                    if idx + 1 >= len(filters) - 1:
                        filters.pop(idx)
                        break
                    if filters[idx + 1] in ['and', 'or']:
                        filters.pop(idx)
                        break
                idx += 1
        if len(filters) > 0:
            where_clause = 'WHERE ' + ' '.join(filters).strip()
            where_clause = where_clause.replace('not_in', 'NOT IN')
        else:
            where_clause = ''

    return where_clause, having_clause


def _predict_group_by(select_clause_str, lst_select_item, order_clause):
    """predict group by according to select and order info

    Args:
        select_clause_str (TYPE): NULL
        lst_select_item (TYPE): NULL
        order_clause (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    agg_count = 0
    for sel in lst_select_item:
        for agg in g_agg_tag_list:
            if agg in sel:
                agg_count += 1
    agg_flag = len(lst_select_item) != agg_count
    
    for agg in g_agg_tag_list:
        if (len(lst_select_item) > 1 and agg in select_clause_str and agg_flag) \
                or agg in order_clause:
            return True
    return False


def _gen_groupby_one_table(sql_json, current_table, select_clause_str, N_T, table_names, pre_table_names):
    """generate 'group by' in case of len(table_names) == 1

    Args:
        sql_json (TYPE): NULL
        current_table (TYPE): NULL
        select_clause_str (TYPE): NULL
        N_T (TYPE): NULL
        table_names (TYPE): NULL
        pre_table_names (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    group_by_clause = ''
    # check none agg
    is_agg_flag = False
    for (magg, mathagg, agg, col, tab) in sql_json['select']:
        if agg == 'none':
            group_by_clause = 'GROUP BY ' + col_to_str(agg, col, tab, table_names, N_T)
        else:
            is_agg_flag = True

    if not is_agg_flag and len(group_by_clause) > 0:
        group_by_clause = "GROUP BY"
        for (magg, mathagg, agg, col, tab) in sql_json['select']:
            group_by_clause = group_by_clause + ' ' + col_to_str(agg, col, tab, table_names, N_T)

    if len(group_by_clause) > 0:
        return group_by_clause

    if 'count(*)' in select_clause_str:
        for primary in current_table['primary_keys']:
            if current_table['table_names'][current_table['col_table'][primary]] != table_names[0]:
                continue
            group_by_clause = 'GROUP BY '
            group_by_clause += col_to_str('none', 
                                          current_table['schema_content'][primary],
                                          current_table['table_names'][current_table['col_table'][primary]],
                                          table_names,
                                          N_T)
            break
    return group_by_clause


def _gen_groupby_one_select(sql_json, current_table, select_clause_str, N_T, table_names, pre_table_names):
    """generate 'group by' in case of len(select) == 1

    Args:
        sql_json (TYPE): NULL
        current_table (TYPE): NULL
        select_clause_str (TYPE): NULL
        N_T (TYPE): NULL
        table_names (TYPE): NULL
        pre_table_names (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    group_by_clause = ''

    magg, mathagg, agg, col, tab = sql_json['select'][0]
    fix_flag = False

    other_tab = None
    for o_tab in table_names.keys():
        if o_tab != tab:
            other_tab = o_tab
    if other_tab:
        for pair in current_table['foreign_keys']:
            t1 = current_table['table_names'][current_table['col_table'][pair[0]]]
            t2 = current_table['table_names'][current_table['col_table'][pair[1]]]
            if t1 in [tab, other_tab] and t2 in [tab, other_tab]:
                if pre_table_names and t1 not in pre_table_names:
                    assert t2 in pre_table_names
                    t1 = t2
                group_by_clause = 'GROUP BY ' + col_to_str('none',
                                                           current_table['schema_content'][pair[0]],
                                                           t1,
                                                           table_names, N_T)
                fix_flag = True
                break

    if not fix_flag:
        group_by_clause = 'GROUP BY ' + col_to_str(agg, col, tab, table_names, N_T)
    return group_by_clause


def _generate_groupby_str(sql_json, current_table, select_clause_str, N_T, table_names, pre_table_names):
    """generate 'group by' string according to other keywords' info

    Args:
        sql_json (TYPE): NULL
        current_table (TYPE): NULL
        select_clause_str (TYPE): NULL
        N_T (int): NULL
        table_names (dict): [in/out]
        pre_table_names (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    group_by_clause = ''
    if len(table_names) == 1:
        return _gen_groupby_one_table(sql_json, current_table, select_clause_str, N_T, table_names, pre_table_names)

    # if only one select
    if len(sql_json['select']) == 1:
        return _gen_groupby_one_select(sql_json, current_table, select_clause_str, N_T, table_names, pre_table_names)

    # check if there are only one non agg
    non_agg, non_agg_count = None, 0
    non_list = []
    for (magg, mathagg, agg, col, tab) in sql_json['select']:
        if agg == 'none':
            non_agg = (agg, col, tab)
            tab_name = ''.join(tab)
            if tab_name not in non_list:
                non_list.append(tab_name)
            non_agg_count += 1

    if non_agg_count == 1:
        return 'GROUP BY ' + col_to_str(non_agg[0], non_agg[1], non_agg[2], table_names, N_T)
    elif non_agg is None:
        return group_by_clause

    fix_flag = False
    if len(non_list) <= 1:
        for key in table_names.keys():
            if key not in non_list:
                non_list.append(key)
    if len(non_list) > 1:
        a = non_list[0]
        b = None
        for non in non_list:
            if a != non:
                b = non
        if b:
            for pair in current_table['foreign_keys']:
                t1 = current_table['table_names'][current_table['col_table'][pair[0]]]
                t2 = current_table['table_names'][current_table['col_table'][pair[1]]]
                if t1 in [a, b] and t2 in [a, b]:
                    if pre_table_names and t1 not in pre_table_names:
                        assert  t2 in pre_table_names
                        t1 = t2
                    group_by_clause = 'GROUP BY ' + col_to_str('none',
                                                               current_table['schema_content'][pair[0]],
                                                               t1,
                                                               table_names, N_T)
                    fix_flag = True
                    break
    tab = non_agg[2]
    assert ''.join(tab) in current_table['table_names']
    if fix_flag:
        return group_by_clause

    find_primary = None
    for primary in current_table['primary_keys']:
        if current_table['table_names'][current_table['col_table'][primary]] == tab:
            find_primary = (current_table['schema_content'][primary], tab)
            return 'GROUP BY ' + col_to_str('none', find_primary[0], find_primary[1], table_names, N_T)

    # rely on count *
    foreign = []
    for pair in current_table['foreign_keys']:
        if current_table['table_names'][current_table['col_table'][pair[0]]] == tab:
            foreign.append(pair[1])
        if current_table['table_names'][current_table['col_table'][pair[1]]] == tab:
            foreign.append(pair[0])

    for pair in foreign:
        if current_table['table_names'][current_table['col_table'][pair]] in table_names:
            group_by_clause = 'GROUP BY ' + col_to_str('none', 
                    current_table['schema_content'][pair],
                    current_table['table_names'][current_table['col_table'][pair]],
                    table_names, N_T)
            return group_by_clause

    for (root_agg, math_agg, agg, col, tab) in sql_json['select']:
        if 'id' in col.lower():
            group_by_clause = 'GROUP BY ' + col_to_str(agg, col, tab, table_names, N_T)
            return group_by_clause

    if len(group_by_clause) <= 5:
        logging.warn('generate group_by clause failed: %s', sql_json['sql']['question_id'])
        #raise RuntimeError('fail to convert')
    return group_by_clause


def to_str(sql_json, N_T, schema, pre_table_names=None):
    """to str
    Args:
        sql_json(dict): sql
        N_T 
        schema(dict):
        pre_table_names:Default is None
    Returns: str
    Rsises: NULL
    """
    all_columns = list()
    select_clause = list()
    table_names = dict()
    current_table = schema

    select_clause_str, lst_select_item = _select_to_str(sql_json['select'], N_T, table_names, all_columns)
    order_clause = ''
    if 'sup' in sql_json:
        order_clause = _order_to_str(sql_json['sup'], N_T, table_names, all_columns, need_limit=True)
    elif 'order' in sql_json:
        order_clause = _order_to_str(sql_json['order'], N_T, table_names, all_columns)

    where_clause, having_clause = _condition_to_str(
            sql_json.get('where', []), N_T, sql_json['sql'], schema, table_names, all_columns)

    group_by_clause = ''
    has_group_by = False
    if len(having_clause) > 0:
        has_group_by = True
    else:
        has_group_by = _predict_group_by(select_clause_str, lst_select_item, order_clause)
    if has_group_by:
        group_by_clause = _generate_groupby_str(
                sql_json, current_table, select_clause_str, N_T, table_names, pre_table_names)

    intersect_clause = ''
    if 'intersect' in sql_json:
        sql_json['intersect']['sql'] = sql_json['sql']
        intersect_clause = 'INTERSECT ' + to_str(sql_json['intersect'], len(table_names) + 1, schema, table_names)
    union_clause = ''
    if 'union' in sql_json:
        sql_json['union']['sql'] = sql_json['sql']
        union_clause = 'UNION ' + to_str(sql_json['union'], len(table_names) + 1, schema, table_names)
    except_clause = ''
    if 'except' in sql_json:
        sql_json['except']['sql'] = sql_json['sql']
        except_clause = 'EXCEPT ' + to_str(sql_json['except'], len(table_names) + 1, schema, table_names)

    # TODO: delete useless code!!!!!!!! 
    table_names_replace = {}
    for a, b in zip(current_table['table_names'], current_table['table_names']):
        table_names_replace[b] = a
    new_table_names = {}
    for key, value in table_names.items():
        if key is None:
            continue
        new_table_names[table_names_replace[key]] = value
    from_clause = infer_from_clause(new_table_names, schema, all_columns).strip()

    lst_all_sub_clause = [select_clause_str, from_clause, where_clause, group_by_clause, having_clause,
                          order_clause, intersect_clause, union_clause, except_clause]
    sql = ' '.join([x for x in lst_all_sub_clause if x.strip() != ''])

    return sql


def _extract_select_col(sql_json):
    """extract select column

    Args:
        sql_json (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    assert len(sql_json['select']) > 0
    (magg, mathagg, agg, col, tab) = sql_json['select'][0]
    return col


def to_str_main(sql_json, N_T, schema, pre_table_names=None):
    """struct query to sql string

    Args:
        sql_json (TYPE): NULL
        N_T (TYPE): NULL
        schema (TYPE): NULL
        pre_table_names (TYPE): Default is None

    Returns: TODO

    Raises: NULL
    """
    row_calc_op = sql_json.get('row_calc_op', None)
    if row_calc_op is None:
        return to_str(sql_json, N_T, schema, pre_table_names)

    row_calc_left = sql_json['row_calc_left']
    row_calc_right = sql_json['row_calc_right']

    calc_col = _extract_select_col(row_calc_left)
    subsql_left = to_str(row_calc_left, N_T, schema, pre_table_names)
    subsql_right = to_str(row_calc_right, N_T, schema, pre_table_names)
    return 'select a.%s %s b.%s from (%s) a, (%s) b' % (calc_col, row_calc_op, calc_col, subsql_left, subsql_right)


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--data-path', required=True, help='train/dev/test data json')
    arg_parser.add_argument('-t', '--table-path', required=True, help='data tables path')
    arg_parser.add_argument('-g', '--grammar-path', required=True, help='grammar path')
    arg_parser.add_argument('-r', '--rule-path', required=True, help='predict rule id path')
    arg_parser.add_argument('-m', '--max-len', type=int, nargs=3, required=False, default=(12, 40, 60),
                                                    help='max len of table column and value')
    arg_parser.add_argument('-o', '--output-path', required=True, help='output data')
    arg_parser.add_argument('--debug', action="store_true", default=False, help='open debug mode')
    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.INFO,
        format='%(levelname)s: %(asctime)s %(filename)s'
        ' [%(funcName)s:%(lineno)d][%(process)d] %(message)s',
        datefmt='%m-%d %H:%M:%S',
        filename=None,
        filemode='a')

    # loading dataSets
    lst_data, schemas = grammar_utils.load_datasets(
            args.data_path, args.table_path, args.grammar_path, args.rule_path, args.max_len)
    grammar_utils.alter_not_in(lst_data, schemas=schemas)

    index = range(len(lst_data))
    count = 0
    exception_count = 0
    with open(args.output_path, 'w', encoding='utf8') as ofs:
        for dct_data in lst_data:
            try:
                result = transform(dct_data, schemas[dct_data['db_id']])
                ofs.write(dct_data['question_id'] + '\t' + result[0] + '\t' + dct_data['db_id'] + '\n')
                count += 1
            except Exception as e:
                if args.debug:
                    traceback.print_exc()
                    exit(-1)
                result = transform(dct_data, schemas[dct_data['db_id']], origin=DFT_RULE)
                exception_count += 1
                ofs.write(dct_data['question_id'] + '\t' + result[0] + '\t' + dct_data['db_id'] + '\n')
                count += 1
                logging.error('%s parse error', dct_data.get('question_id', 'rule'))

    logging.info('total: %d. error: %d', count, exception_count)
