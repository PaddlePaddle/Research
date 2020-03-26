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
"""text2sql"""

class DName(object):

    """Data Name Constants"""

    INPUT_IDS = 'INPUT_IDS'
    MASK_IDS = 'MASK_IDS'
    SEQ_LENS = 'SEQ_LENS'
    NAME_LENS = 'NAME_LENS'
    NAME_POS = 'NAME_POS'
    NAME_TOK_LEN = 'NAME_TOK_LEN'
    NAME_BEGIN = 'NAME_BEGIN'
    NAME_END = 'NAME_END'

    Q_POS = 'QUESTION_POS'
    T_POS = 'TABLE_NAME_POS'
    C_POS = 'COLUMN_NAME_POS'
    V_POS = 'VALUE_POS'
    Q_LEN = 'QUESTION_LENS'
    T_LEN = 'TABLE_LENS'
    C_LEN = 'COLUMN_LENS'
    V_LEN = 'VALUE_LENS'
    Q_SPAN_LEN = 'QUESTION_SPAN_LENS'
    T_TOKS_LEN = 'TABLE_TOKENS_LENS'
    C_TOKS_LEN = 'COLUMN_TOKENS_LENS'
    V_TOKS_LEN = 'VALUE_TOKENS_LENS'

    TRAIN_LABEL = 'TRAIN_LABEL'
    INFER_LABEL = 'INFER_LABEL'
    INFER_ACTIONS = 'INFER_ACTIONS'
    INFER_VOCAB_MASK = 'INFER_VOCAB_MASK'
    INFER_GMR_MASK = 'INFER_GMR_MASK'
    INFER_COL2TABLE_MASK = 'INFER_COL2TABLE_MASK'

    QTC_IDS = 'QTC_IDS'
    QTC_SENTENCE_IDS = 'QTC_SENTENCE_IDS'
    QTC_POS_IDS = 'QTC_POS_IDS'
    QTC_MASK_IDS = 'QTC_MASK_IDS'
    QTC_TASK_IDS = 'QTC_TASK_IDS'
    QTC_SEQ_LENS = 'QTC_SEQ_LENS'
    QV_IDS = 'QV_IDS'
    QV_SENTENCE_IDS = 'QV_SENTENCE_IDS'
    QV_POS_IDS = 'QV_POS_IDS'
    QV_MASK_IDS = 'QV_MASK_IDS'
    QV_TASK_IDS = 'QV_TASK_IDS'
    QV_SEQ_LENS = 'QV_SEQ_LENS'


from text2sql.framework import register

register.import_new_module("text2sql.datalib", "json_dataset")
register.import_new_module("text2sql.datalib", "text1d_field_reader")
register.import_new_module("text2sql.datalib", "text2d_field_reader")
register.import_new_module("text2sql.datalib", "feature_field_reader")
register.import_new_module("text2sql.datalib", "ernie_field_reader")
register.import_new_module("text2sql.datalib", "grammar_label_field_reader")

