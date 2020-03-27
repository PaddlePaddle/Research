# -*- coding:utf8 -*-
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
"""grammar model"""

from collections import namedtuple

DecoderInputsWrapper = namedtuple("DecoderInputsWrapper", "input action gmr_mask")
DecoderDynamicVocab = namedtuple("DecoderDynamicVocab",
                                 "table table_len column column_len value value_len column2table_mask")

from text2sql.models.grammar.nets import grammar_output
from text2sql.models.grammar.infer_decoder import GrammarInferDecoder
from text2sql.models.grammar.dynamic_decode import decode_with_grammar

