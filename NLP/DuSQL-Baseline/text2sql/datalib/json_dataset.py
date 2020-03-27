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

"""json dataset, and it's corresponding reader
"""

import sys
import os
import traceback
import logging
import json
from collections import namedtuple

from text2sql.framework import register
from text2sql.framework.reader.data_set_reader.basic_dataset_reader import BasicDataSetReader
from text2sql.framework.reader.data_set import DataSet

@register.RegisterSet.data_set_reader.register
class JsonDataSetReader(BasicDataSetReader):

    """JsonDataSetReader. """

    def __init__(self, name, fields, config):
        """init of class

        Args:
            name (TYPE): NULL
            fields (TYPE): NULL
            config (TYPE): NULL
        """
        super(JsonDataSetReader, self).__init__(name, fields, config)

    def read_files(self, file_path, quotechar=None):
        """read data from a json file

        Args:
            file_path (TYPE): NULL
            quotechar (TYPE): default is None

        Returns: TODO

        Raises: NULL

        """
        with open(file_path) as ifs:
            lst_data_in = json.load(ifs)

        lst_field_name = [field.name for field in self.fields]
        Example = namedtuple("Example", lst_field_name)
        lst_all_example = []
        for data_item in lst_data_in:
            lst_data_extracted = []
            for name in lst_field_name:
                curr_data = data_item[name]
                if name == "question_tokens" and type(curr_data[0]) is not list:
                    curr_data = ' '.join(curr_data)
                elif name in ("question_tokens", "table_names", "column_names", "values"):
                    curr_data = ' [SEP] '.join([' '.join(x) for x in curr_data])
                elif name in ("label"):
                    curr_data = curr_data
                elif name.endswith("_features") or name == 'column_tables':
                    curr_data = ' [SEP] '.join([' '.join([str(y) for y in x]) for x in curr_data])
                lst_data_extracted.append(curr_data)
            example = Example(*lst_data_extracted)
            lst_all_example.append(example)

        return lst_all_example

    def serialize_batch_records(self, batch_records):
        """re-organize batch records

        Args:
            batch_records (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        field_names = batch_records[0]._fields
        has_label = field_names[-1] == 'label'

        records_split_fields = list(zip(*batch_records))
        batch_question, batch_tname, batch_cname, batch_value, \
                batch_q_fea, batch_t_fea, batch_c_fea, batch_v_fea, \
                batch_column2tables = records_split_fields[:9]
        batch_label = None
        if has_label:
            batch_label = records_split_fields[9]

        question_result = self._cvt_data(field_names, 'question_tokens', batch_question)
        tname_result = self._cvt_data(field_names, 'table_names', batch_tname)
        cname_result = self._cvt_data(field_names, 'column_names', batch_cname)
        value_result = self._cvt_data(field_names, 'values', batch_value)
        qfea_result  = self._cvt_data(field_names, 'question_features', batch_q_fea)
        tfea_result  = self._cvt_data(field_names, 'table_features', batch_t_fea)
        cfea_result  = self._cvt_data(field_names, 'column_features', batch_c_fea)
        vfea_result  = self._cvt_data(field_names, 'value_features', batch_v_fea)
        col2tbl_result  = self._cvt_data(field_names, 'column_tables', batch_column2tables)
        label_result = None
        if has_label:
            label_result = self._cvt_data(
                        field_names,
                        'label',
                        (batch_label, batch_tname, batch_cname, batch_value, col2tbl_result[0]))

        final_result = question_result + tname_result + cname_result + value_result + \
                       qfea_result + tfea_result + cfea_result + vfea_result + col2tbl_result
        if has_label:
            final_result += label_result
        return final_result

    def _cvt_data(self, field_names, name, data_in):
        """convert data to id
        Args:
            field_names (tuple): name of each fields in order
            name (str): data name
            data_in (ndarray/tuple): params for convert_texts_to_ids
        Returns: TODO
        """
        idx = field_names.index(name)
        return self.fields[idx].field_reader.convert_texts_to_ids(data_in)


@register.RegisterSet.data_set_reader.register
class ErnieJsonDataSetReader(JsonDataSetReader):

    """ErnieJsonDataSetReader. """

    def __init__(self, name, fields, config):
        """init of class

        Args:
            name (TYPE): NULL
            fields (TYPE): NULL
            config (TYPE): NULL
        """
        super(ErnieJsonDataSetReader, self).__init__(name, fields, config)

    def serialize_batch_records(self, batch_records):
        """serialize batch records

        Args:
            batch_records (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        field_names = batch_records[0]._fields
        has_label = field_names[-1] == 'label'

        records_split_fields = list(zip(*batch_records))
        batch_question, batch_tname, batch_cname, batch_value, \
                batch_q_fea, batch_t_fea, batch_c_fea, batch_v_fea, \
                batch_column2tables = records_split_fields[:9]
        batch_label = None
        if has_label:
            batch_label = records_split_fields[9]

        input_all = self._cvt_data(field_names, 'question_tokens',
                                         (batch_question, batch_tname, batch_cname, batch_value))
        qfea_result  = self._cvt_data(field_names, 'question_features', batch_q_fea)
        tfea_result  = self._cvt_data(field_names, 'table_features', batch_t_fea)
        cfea_result  = self._cvt_data(field_names, 'column_features', batch_c_fea)
        vfea_result  = self._cvt_data(field_names, 'value_features', batch_v_fea)
        col2tbl_result  = self._cvt_data(field_names, 'column_tables', batch_column2tables)
        label_result = None
        if has_label:
            label_result = self._cvt_data(
                    field_names,
                    'label',
                    (batch_label, batch_tname, batch_cname, batch_value, col2tbl_result[0]))

        final_result = input_all * 4 + qfea_result + tfea_result + cfea_result + vfea_result + col2tbl_result
        if has_label:
            final_result += label_result
        return final_result


class T2SDataSet(DataSet):

    """text2sql data set"""

    def __init__(self, params_dict):
        """init of class

        Args:
            param_dict (TYPE): NULL
        """
        super(T2SDataSet, self).__init__(params_dict)

        self.build()


if __name__ == "__main__":
    """run some simple test cases"""
    pass

