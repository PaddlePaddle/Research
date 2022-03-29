#!/usr/bin/env python3
# -*- coding:utf-8 -*-
##########################################################
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved #
##########################################################

"""auto generate simple query&question

Filname: auto_gen_simple_query.py
Authors: ZhangAo(@baidu.com)
Date: 2020-06-03 14:34:39
Modified: modified by wukun04 
"""

import sys
import os
import traceback
import logging
import json
import random
import copy
from collections import defaultdict
from nltk.tokenize import word_tokenize

random.seed(9999)
AGG_OPS = ('', 'max', 'min', 'count', 'sum', 'avg')
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
WHERE_OPS_NL = ('not', 'between', 'equal to', 'more than', 'less than', 'no less than', 'no more than', 'not equal to', 'in', 'like', 'is', 'exists')

logging.basicConfig(level=logging.DEBUG,
        format='%(levelname)s: %(asctime)s %(filename)s'
        ' [%(funcName)s:%(lineno)d][%(process)d] %(message)s',
        datefmt='%m-%d %H:%M:%S',
        filename=None,
        filemode='a')

g_max_conds_num = 3
get_and_or = lambda : ['and', 'and', 'or'][random.randint(0, 2)]

class SelectOneUnit(object):

    """val_unit: (unit_op, col_unit1, col_unit2)
       col_unit: (agg_id, col_id, isDistinct(bool))
    """
    BASE = [0, ['AggID', "ColID", False], None]

    @classmethod
    def create(cls, col_id, agg_type):
        """create a col unit

        Args:
            col_id (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        assert type(col_id) is int, "col_id must be int"

        if agg_type not in ["number"]:
            agg_id = random.choice([0, 0, 0, 0, 3])
        else:
            agg_id = random.choice([0, 0, 0, 0, 0, 1, 2, 5])
        col_unit = copy.deepcopy(cls.BASE)
        col_unit[1][1] = col_id
        col_unit[1][0] = agg_id
        return col_unit

class ValueUnit(object):

    """val_unit: (unit_op, col_unit1, col_unit2)
       col_unit: (agg_id, col_id, isDistinct(bool))
    """
    BASE = [0, [0, "ColID", False], None]

    @classmethod
    def create(cls, col_id):
        """create a col unit

        Args:
            col_id (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        assert type(col_id) is int, "col_id must be int"

        col_unit = copy.deepcopy(cls.BASE)
        col_unit[1][1] = col_id
        return col_unit

class GroupUnit(object):

    """
       col_unit: (agg_id, col_id, isDistinct(bool))
    """
    BASE = [[0, "ColID", False]]

    @classmethod
    def create(cls, col_id):
        """create a col unit

        Args:
            col_id (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        assert type(col_id) is int, "col_id must be int"

        col_unit = copy.deepcopy(cls.BASE)
        col_unit[0][1] = col_id
        return col_unit

class OrderUnit(object):

    """order_unit: (asc/desc, [col_unit1, col_unit2])
       col_unit: (agg_id, col_id, isDistinct(bool))
    """
    BASE = ['asc', [[0, [0, "ColID", False], None]]]

    @classmethod
    def create(cls, col_id, direct):
        """create a col unit

        Args:
            col_id (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        assert type(col_id) is int, "col_id must be int"
        assert direct in ['asc', 'desc']

        order_unit = copy.deepcopy(cls.BASE)
        order_unit[0] = direct
        order_unit[1][0][1][1] = col_id
        return order_unit


class Condition(object):

    """val_unit: (unit_op, col_unit1, col_unit2)
       col_unit: (agg_id, col_id, isDistinct(bool))
    """
    BASE = [False, 2, "val_unit", 'value', None]

    @classmethod
    def create(cls, col_id, value, op=2):
        """create a col unit

        Args:
            col_id (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        assert type(col_id) is int, "col_id must be int"

        cond = copy.deepcopy(cls.BASE)
        cond[1] = op
        cond[2] = ValueUnit.create(col_id)
        cond[3] = value
        return cond


class ConditionSQL(object):

    """val_unit: (unit_op, col_unit1, col_unit2)
       col_unit: (agg_id, col_id, isDistinct(bool))
       val: sql
    """
    BASE = [False, 2, "val_unit", 'value', None]

    @classmethod
    def create(cls, col_id, table_id, agg=0, op=2):
        """create a col unit

        Args:
            col_id (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        assert type(col_id) is int, "col_id must be int"

        cond = copy.deepcopy(cls.BASE)
        cond[1] = op
        cond[2] = ValueUnit.create(col_id)
        sql_select = [0, [agg, col_id, False], None]
        base_sql = {
            'select': [False, sql_select],
            'from': {"table_units": [["table_unit", table_id]], "conds":[]},
            'where': [],
            'groupBy': [],
            'having': [],
            'orderBy': [],
            'limit': None,
            'intersect': None,
            'union': None,
            'except': None,
        }
        cond[3] = base_sql
        return cond

class HavingUnit(object):

    """val_unit: (unit_op, col_unit1, col_unit2)
       col_unit: (agg_id, col_id, isDistinct(bool))
    """
    BASE = [
            [
                False, 
                3, 
                [
                    0,
                    [
                        3,
                        0,
                        False
                    ],
                    None
                ], 
                'value', 
                None
            ]
        ]

    @classmethod
    def create(cls, value):
        """create a col unit

        Args:
            col_id (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """

        cond = copy.deepcopy(cls.BASE)
        cond[0][3] = value
        return cond


def gen_conds(cond_cols, rows):
    """

    Args:
        cond_cols (TYPE): NULL
        rows (TYPE): NULL

    Returns: (col_id, col_name, col_name_original, value)

    Raises: NULL
    """
    row_idx = random.randint(0, len(rows) - 1)
    active_row = rows[row_idx]
    return [col_info + (active_row[col_info[0]], ) for col_info in cond_cols]

def gen_order(order_col):
    """

    Args:
        cond_cols (TYPE): NULL

    Returns: (col_id, col_name, col_name_original, 'asc'/'desc', limit)

    Raises: NULL
    """
    order = random.random(0, 1) > 0.5
    limit = random.random(0, 1) > 0.5
    asc = random.random(0, 1) > 0.5
    limit_num = 1
    if not order:
        return None
    
    return [order_col[0], order_col[1], order_col[2], 'asc' if asc else 'desc', limit_num if limit else None]



def build_instance(db_id, question, query, query_no_value, sql_json):
    """

    Args:
        db_id (TYPE): NULL
        question (TYPE): NULL
        query (TYPE): NULL
        query_no_value (TYPE): NULL
        sql_json (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    instance = {
            "db_id": db_id,
            "question": question,
            "question_toks": word_tokenize(question),
            "query": query,
            "query_toks": word_tokenize(query),
            "query_toks_no_value": word_tokenize(query_no_value),
            "sql": sql_json
            }
    return instance


def gen_instance(select_cols, other_cols, rows, db_id, table, max_conds):
    """
    Args:
        select_cols (TYPE): [(col_id1, col_name1, col_name1_orignial), (...), ...]
        other_cols (TYPE): [(col_id1, col_name1, col_name1_orignial), (...), ...]
        rows (TYPE): NULL
        db_id (TYPE): NULL
        table (TYPE): (table_id, table_name, table_names_original)
        max_conds (TYPE): NULL

    Returns: list of instance
        instance = {
            db_id: str
            question: str
            question_toks: list of str
            query: str
            query_toks: list of str
            query_toks_no_value: list of str
            sql: {
                select: [False, list of (0, ValueUnit)]
                from: {
                    conds: []
                    table_units: [["table_unit", TableID]]
                }
                where: [Condition, and/or, Condition, ...]
                groupBy: []
                having: []
                orderBy: []
                limit: None
                intersect: None
                union: None
                except: None
            }
        }

        Condition = [False, 2, ValueUnit, Value, None]
        ValueUnit = [0, [0, ColID, False], None]
    Raises: NULL
    """
    
    base_val_unit = [0, [0, "ColID", False], None]
    sql_select = [SelectOneUnit.create(x[0], x[3]) for x in select_cols]
    agg_select = [AGG_OPS[sel[1][0]] for sel in sql_select]
    #print(agg_select)
    base_sql = {
            'select': [False, sql_select],
            'from': {"table_units": [["table_unit", table[0]]], "conds":[]},
            'where': [],
            'groupBy': [],
            'having': [],
            'orderBy': [],
            'limit': None,
            'intersect': None,
            'union': None,
            'except': None,
        }
    question_select = "Show " + ' and '.join('the ' + x[1] for  x, agg in zip(select_cols, agg_select))
    query_select_from = 'SELECT ' + ' , '.join(f"{agg.upper()} ( {x[2]} )" if agg else x[2] for x, agg in zip(select_cols, agg_select)) + ' FROM ' + table[2]
    query_nv_select_from = query_select_from

    query_sub = 'SELECT ' + ' , '.join(f"{agg.upper()} {x[2]}" if agg else x[2] for x, agg in zip(select_cols, agg_select)) + ' FROM ' + table[2]

    # no condition
    if len(rows) == 0:
        sub_sql = {}
        sub_sql['select'] = query_sub.lower()
        sql = copy.deepcopy(base_sql)
        sub_sql_list = [sub_sql['select'].lower().replace('_', ' ')]
        instance = build_instance(db_id, '', query_select_from, query_nv_select_from, sql)
        instance['sub_sql_dict'] = sub_sql
        instance['sub_sql_list'] = sub_sql_list
        return [instance]

    instances = []
    for cond_num in range(0, min(max_conds + 1, len(rows))):
        sub_sql = {}
        sub_sql['select'] = query_sub
        if len(other_cols) < cond_num:
            break
        cond_cols = random.sample(other_cols, k=cond_num)
        conds = gen_conds(cond_cols, rows)
        sql = copy.deepcopy(base_sql)
        question_conds = []
        query_conds = []
        query_nv_conds = []
        query_sub_conds = []
        cond_conn = get_and_or()


        ###### conds
        x = random.random() 
        if x > 0.9 and cond_num == 1 and conds[0][3] == 'number':
            op = random.choice([ 2, 2, 2, 2, 3, 4])
            agg = random.choice([1, 2, 5])
            col_id = conds[0][0]
            cond = ConditionSQL.create(col_id, table[0], agg, op)
            sql['where'].append(cond)
            query_conds = ' where ' + f'{conds[0][2]} {WHERE_OPS[op]} ( SELECT {AGG_OPS[agg]} ( {conds[0][2]} ) FROM {table[2]} )'
            query_nv = query_select_from + query_conds
            query_sub = 'where ' + f'{conds[0][2]} {WHERE_OPS_NL[op]} ( SELECT {AGG_OPS[agg]} ( {conds[0][2]} ) FROM {table[2]} )'
            query = query_select_from + query_conds
            sub_sql['where'] = query_sub.lower()
        else:
            for col_id, col_name, col_name_original, ctype, value in conds:
                if len(sql['where']) > 0:
                    sql['where'].append(cond_conn)
                
                op = random.choice([2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7]) if ctype in ['number'] else random.choice([2, 2, 2, 2, 7, 9]) 
                sql['where'].append(Condition.create(col_id, value, op=op))
                question_conds.append('%s %s %s' % (col_name, WHERE_OPS[op], value))
                query_conds.append('%s %s "%s"' % (col_name_original, WHERE_OPS[op], value))
                query_nv_conds.append('%s %s value' % (col_name_original, WHERE_OPS[op]))
                query_sub_conds.append('%s %s %s' % (col_name, WHERE_OPS_NL[op], value))

            if len(conds) > 0:
                cond_conn = ' ' + cond_conn + ' '
                question = question_select + ' when ' + cond_conn.join(question_conds)
                query = query_select_from + ' where ' + cond_conn.join(query_conds)
                query_nv = query_select_from + ' where ' + cond_conn.join(query_nv_conds)
                query_sub = ' where ' + cond_conn.join(query_sub_conds)

                sub_sql['where'] = query_sub
            else:
                query = query_select_from
                query_nv = query_select_from

        # instance = build_instance(db_id, "", query, query_nv, sql)
        # instance['sub_sql_dict'] = sub_sql
        # instances.append(instance)

        ##### ORDER BY
        if cond_num <= 1:
            sql = copy.deepcopy(sql)
            order_col = random.choice(other_cols)
            if order_col in cond_cols:
                pass
            else:
                sql['orderBy'] = OrderUnit.create(order_col[0], 'asc' if random.random() > 0.5 else 'desc')
                sql['limit'] = 1 if random.random() > 0.5 else None
                query_order = f"ORDER BY {order_col[1]} {sql['orderBy'][0].upper()}" if not sql['limit'] else f"ORDER BY {order_col[1]} {sql['orderBy'][0].upper()} LIMIT 1"
                query_order_nv = f"ORDER BY {order_col[1]} {sql['orderBy'][0].upper()}" if not sql['limit'] else f"ORDER BY {order_col[1]} {sql['orderBy'][0].upper()} LIMIT value"
                sub_sql['orderBy'] = query_order
                query_nv = query_nv + ' ' + query_order_nv
                query = query + ' ' + query_order 

                # instance = build_instance(db_id, "", query, query_nv, sql)
                # instance['sub_sql_dict'] = sub_sql
                # instances.append(instance)
            ####### GROUP BY
            if random.random() > 0.6:
                sql = copy.deepcopy(sql)
                group_col = random.choice(other_cols)
                if group_col in cond_cols or group_col == order_col:
                    pass
                else:
                    sql['groupBy'] = GroupUnit.create(group_col[0])
                    query_group = f"GROUP BY {group_col[1]}"
                    query_group_nv = query_group
                    sub_sql['groupBy'] = query_group
                    query_nv = query_nv + ' ' + query_group_nv
                    query = query + ' ' + query_group

                    # instance = build_instance(db_id, "", query, query_nv, sql)
                    # instance['sub_sql_dict'] = sub_sql
                    # instances.append(instance)
                    
                    #### HAVING
                    if random.random() > 0.7:
                        sql = copy.deepcopy(sql)
                        sql['having'] = HavingUnit.create(1.0)
                        query_having = f"HAVING count * > {1}"
                        query_having_sub = f"HAVING count all {WHERE_OPS_NL[3]} {1}"
                        query_having_nv = f"HAVING count(*) > value"
                        sub_sql['groupBy'] = query_having_sub
                        query_nv = query_nv + ' ' + query_having_nv
                        query = query + ' ' + query_having

                        # instance = build_instance(db_id, "", query, query_nv, sql)
                        # instance['sub_sql_dict'] = sub_sql
                        # instances.append(instance)
        instance = build_instance(db_id, "", query, query_nv, sql)
        instance['sub_sql_dict'] = sub_sql
        sub_sql_list = []
        if 'where' in sub_sql:
            sub_sql_list.append(sub_sql['where'].lower().replace('_', ' '))
        if 'select' in sub_sql:
            sub_sql_list.append(sub_sql['select'].lower().replace('_', ' '))
        if 'groupBy' in sub_sql and 'orderBy' in sub_sql:
            sub_sql_list.append(sub_sql['groupBy'].lower().replace('_', ' ') + ' ' +sub_sql['orderBy'].lower().replace('_', ' '))
        else:
            if 'groupBy' in sub_sql:
                sub_sql_list.append(sub_sql['groupBy'].lower().replace('_', ' '))
            if 'orderBy' in sub_sql:
                sub_sql_list.append(sub_sql['orderBy'].lower().replace('_', ' '))
        instance['sub_sql_list'] = sub_sql_list
        instances.append(instance)

    return instances


def auto_gen_one_table(db_id, table, schema, rows, ofs):
    """

    Args:
        db_id (TYPE): NULL
        table (TYPE): (table_id, table_name)
        schema (TYPE): [(col_name, col_name_original), ...]
        rows (TYPE): NULL
        ofs (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    lst_result = []
    for _ in range(9):
        for k in range(1, 4):
            cols = [(col_id,) + col_name_info for col_id, col_name_info in enumerate(schema)]
            select_cols = random.sample(cols, min(k, len(cols)))
            select_idxs = [item[0] for item in select_cols]
            other_cols = [(idx, ) + col for idx, col in enumerate(schema) if idx not in select_idxs]
            random.shuffle(other_cols)
            try:
                instances = gen_instance(select_cols,
                                    other_cols,
                                    rows,
                                    db_id,
                                    table,
                                    max_conds = g_max_conds_num)
                lst_result.extend(instances)
            except:
                pass
    return lst_result


def auto_gen(db_id, tables, contents, ofs):
    """

    Args:
        db_id (TYPE): NULL
        header (TYPE): NULL
        rows (TYPE): NULL
        ofs (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    dct_table2schema = defaultdict(list)
    table_names = tables['table_names']
    table_names_original = tables['table_names_original']
    for (tid, column), (_, col_original), col_type in zip(tables['column_names'], tables['column_names_original'], tables["column_types"]):
        if tid == -1:
            continue
        dct_table2schema[(tid, table_names[tid], table_names_original[tid])].append((column, col_original, col_type))
    
    lst_results = []
    for (tid, tname, tname_original), schema in dct_table2schema.items():
        rows = []
        if tname_original.lower() not in contents['tables']:
            continue
        for x in contents['tables'][tname_original.lower()].get('values', []):
            try:
                rows.append(eval(x.replace('\n', '\\n').replace('\r', '\\r')))
            except:
                continue
        table_names_original = tables['table_names_original'][tid]
        lst_result = auto_gen_one_table(db_id, (tid, tname, table_names_original), schema, rows, ofs)
        lst_results.extend(lst_result)
        
    return lst_results

if __name__ == "__main__":
    import argparse
    try:
        arg_parser = argparse.ArgumentParser(description="auto generate simple query&question")
        # arg_parser.add_argument("input", nargs="?", type=argparse.FileType('r'), default=sys.stdin,
        #                         help="input file path")
        arg_parser.add_argument("-i", "--input", type=argparse.FileType('r'),
                                help="input file path")
        arg_parser.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout,
                                help="output file path")
        arg_parser.add_argument("-d", "--dbid-set", type=argparse.FileType('r'), help="db id set file path")
        arg_parser.add_argument("-t", "--db-content", type=argparse.FileType('r'), help="db content file path")
        arg_parser.add_argument("-c", "--max-conds", default=g_max_conds_num, type=int, help='max condition number')
        arg_parser.add_argument("--seed", default=1, type=int, help='random seed')
        args = arg_parser.parse_args()

        random.seed(args.seed)
        g_max_conds_num = args.max_conds
        g_allow_db_set = set(x.strip() for x in args.dbid_set)
        g_dbid2content = {x['db_id']: x for x in json.load(args.db_content)}

        lst_results = []
        for tables in json.load(args.input):
            db_id = tables['db_id']
            if db_id not in g_allow_db_set:
                continue
            if db_id not in g_dbid2content:
                continue
            contents = g_dbid2content[db_id]

            lst_result = auto_gen(db_id, tables, contents, args.output)
            lst_results.extend(lst_result)
        for item in lst_results:
            args.output.write(json.dumps(item, ensure_ascii=False) + '\n')
            #json.dump(lst_results, args.output, indent=4, ensure_ascii=False)
    except Exception as e:
        traceback.print_exc()
        #logging.critical(traceback.format_exc())
        exit(-1)

    print(f'generate {len(lst_results)} cases')