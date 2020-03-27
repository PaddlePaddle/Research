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

"""grammar defination
"""

g_keywords = ['des', 'asc', 'and', 'or', 'sum', 'min', 'max', 'avg', 'none',
              '=', '!=', '<', '>', '<=', '>=', 'between', 'like', 'not_like',
              'in', 'not_in', 'count', 'intersect', 'union', 'except']
g_keywords = set(g_keywords)

class Grammar(object):
    """ Grammar class"""
    def __init__(self, is_sketch=False):
        self.begin = 0
        self.type_id = 0
        self.is_sketch = is_sketch
        self.prod2id = {}
        self.type2id = {}
        self._init_grammar(Select)
        self._init_grammar(SingleSQL)
        self._init_grammar(Superlative)
        self._init_grammar(Filter)
        self._init_grammar(Order)
        self._init_grammar(NumA)
        self._init_grammar(SQL)

        if not self.is_sketch:
            self._init_grammar(Agg)

        self._init_id2prod()
        self.type2id[Column] = self.type_id
        self.type_id += 1
        self.type2id[Table] = self.type_id

    def _init_grammar(self, Cls):
        """
        get the production of class Cls
        :param Cls:
        :return:
        """
        production = Cls._init_grammar()
        for p in production:
            self.prod2id[p] = self.begin
            self.begin += 1
        self.type2id[Cls] = self.type_id
        self.type_id += 1

    def _init_id2prod(self):
        self.id2prod = {}
        for key, value in self.prod2id.items():
            self.id2prod[value] = key

    def get_production(self, Cls):
        """ get production
            Args: 
                Cls: cls

            Return: TODO
            Raises: NULL 
        
        """
        return Cls._init_grammar()


class Action(object):
    """ action class
        object
    
    """
    def __init__(self):
        self.pt = 0
        self.production = None
        self.children = list()

    def get_next_action(self, is_sketch=False):
        """ get next action
        Args:
            is_sketch(TYPE): Default is False

        Returns: TODO
        Raises: NULL
        """
        actions = list()
        for x in self.production.split(' ')[1:]:
            if x not in g_keywords:
                rule_type = eval(x)
                if is_sketch:
                    if rule_type is not Agg:
                        actions.append(rule_type)
                else:
                    actions.append(rule_type)
        return actions

    def set_parent(self, parent):
        """ set parent
        Args: 
            parent : parent action
        Returns: NULL
        Raises: NULL
        """
        self.parent = parent

    def add_children(self, child):
        """add children
        Args: 
            child: child action
        Returns: NULL
        Raises: NULL
        """
        self.children.append(child)


class SQL(Action):
    """ 
    SQL
    """
    def __init__(self, id_c, parent=None):
        super(SQL, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        # TODO: should add Root grammar to this
        self.grammar_dict = {
            0: 'SQL intersect SingleSQL SingleSQL END',
            1: 'SQL union SingleSQL SingleSQL END',
            2: 'SQL except SingleSQL SingleSQL END',
            3: 'SQL + SingleSQL SingleSQL END',
            4: 'SQL - SingleSQL SingleSQL END',
            5: 'SQL * SingleSQL SingleSQL END',
            6: 'SQL / SingleSQL SingleSQL END',
            7: 'SQL SingleSQL END',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'SQL(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'SQL(' + str(self.id_c) + ')'


class SingleSQL(Action):
    """ 
    SingleSQL
    """
    def __init__(self, id_c, parent=None):
        super(SingleSQL, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        # TODO: should add SingleSQL grammar to this
        self.grammar_dict = {
            0: 'SingleSQL Select Superlative Filter',
            1: 'SingleSQL Select Filter Order',
            2: 'SingleSQL Select Superlative',
            3: 'SingleSQL Select Filter',
            4: 'SingleSQL Select Order',
            5: 'SingleSQL Select'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'SingleSQL(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'SingleSQL(' + str(self.id_c) + ')'


class NumA(Action):
    """
    Number of Columns
    """
    def __init__(self, id_c, parent=None):
        super(NumA, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'NumA Agg',
            1: 'NumA Agg Agg',
            2: 'NumA Agg Agg Agg',
            3: 'NumA Agg Agg Agg Agg',
            4: 'NumA Agg Agg Agg Agg Agg'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'NumA(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'NumA(' + str(self.id_c) + ')'


class Agg(Action):
    """
    Aggregator
    """
    def __init__(self, id_c, parent=None):
        super(Agg, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        # TODO: should add SingleSQL grammar to this
        self.grammar_dict = {
            0: 'Agg none Column Table',
            1: 'Agg max Column Table',
            2: "Agg min Column Table",
            3: "Agg count Column Table",
            4: "Agg sum Column Table",
            5: "Agg avg Column Table",
            6: "Agg none MathAgg",
            7: "Agg max MathAgg",
            8: "Agg min MathAgg",
            9: "Agg count MathAgg",
            10: "Agg sum MathAgg",
            11: "Agg avg MathAgg"
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Agg(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Agg(' + str(self.id_c) + ')'


class MathAgg(Action):
    """
    MathaAgg
    """
    def __init__(self, id_c, parent=None):
        super(MathAgg, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        # TODO: should add SingleSQL grammar to this
        self.grammar_dict = {
            0: 'MathAgg + Agg Agg',
            1: 'MathAgg - Agg Agg',
            2: "MathAgg * Agg Agg",
            3: "MathAgg / Agg Agg",
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'MathAgg(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'MathAgg(' + str(self.id_c) + ')'


class Select(Action):
    """
    Select
    """
    def __init__(self, id_c, parent=None):
        super(Select, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Select NumA'
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Select(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Select(' + str(self.id_c) + ')'

class Filter(Action):
    """
    Filter
    """
    def __init__(self, id_c, parent=None):
        super(Filter, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            # 0: "Filter 1"
            0: 'Filter and Filter Filter',
            1: 'Filter or Filter Filter',
            2: 'Filter = Agg Value',
            3: 'Filter != Agg Value',
            4: 'Filter < Agg Value',
            5: 'Filter > Agg Value',
            6: 'Filter <= Agg Value',
            7: 'Filter >= Agg Value',
            8: 'Filter between Agg Value Value',
            9: 'Filter like Agg Value',
            10: 'Filter not_like Agg Value',
            # now begin root
            11: 'Filter = Agg SingleSQL',
            12: 'Filter < Agg SingleSQL',
            13: 'Filter > Agg SingleSQL',
            14: 'Filter != Agg SingleSQL',
            15: 'Filter between Agg SingleSQL SingleSQL',
            16: 'Filter >= Agg SingleSQL',
            17: 'Filter <= Agg SingleSQL',
            # now for In
            18: 'Filter in Agg SingleSQL',
            19: 'Filter not_in Agg SingleSQL'

        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Filter(' + str(self.id_c) + ')'

    def __repr__(self):
#        return 'Filter(' + str(self.grammar_dict[self.id_c]) + ')'
        return 'Filter(' + str(self.id_c) + ')'


class Superlative(Action):
    """
    Superlative
    """
    def __init__(self, id_c, parent=None):
        super(Superlative, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Superlative des Agg',
            1: 'Superlative asc Agg',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Superlative(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Superlative(' + str(self.id_c) + ')'


class Order(Action):
    """
    Order
    """
    def __init__(self, id_c, parent=None):
        super(Order, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self._init_grammar()
        self.production = self.grammar_dict[id_c]

    @classmethod
    def _init_grammar(self):
        self.grammar_dict = {
            0: 'Order des Agg',
            1: 'Order asc Agg',
        }
        self.production_id = {}
        for id_x, value in enumerate(self.grammar_dict.values()):
            self.production_id[value] = id_x

        return self.grammar_dict.values()

    def __str__(self):
        return 'Order(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Order(' + str(self.id_c) + ')'


class Column(Action):
    """
    Column
    """
    def __init__(self, id_c, parent=None):
        super(Column, self).__init__()
        self.parent = parent
        self.id_c = id_c
        self.production = 'Column Table'
        self.table = None

    def __str__(self):
        return 'Column(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Column(' + str(self.id_c) + ')'


class Table(Action):
    """
    Table
    """
    def __init__(self, id_c, parent=None):
        super(Table, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self.production = 'Table min'
        self.table = None

    def __str__(self):
        return 'Table(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Table(' + str(self.id_c) + ')'


class Value(Action):
    """
    Value
    """
    def __init__(self, id_c, parent=None):
        super(Value, self).__init__()

        self.parent = parent
        self.id_c = id_c
        self.production = 'Value'
        self.table = None

    def __str__(self):
        return 'Value(' + str(self.id_c) + ')'

    def __repr__(self):
        return 'Value(' + str(self.id_c) + ')'


if __name__ == '__main__':
     print(list(SingleSQL._init_grammar()))

