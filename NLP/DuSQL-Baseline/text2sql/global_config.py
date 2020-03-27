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

"""global config
"""

import sys
import os
import traceback
import logging

class GlobalConfig(object):

    """Global Config for the System """

    def __init__(self):
        """init of class """
        super(GlobalConfig, self).__init__()

        self.use_question_feature = False
        self.use_table_feature = False
        self.use_column_feature = False
        self.use_value_feature = False


if __name__ == "__main__":
    """run some simple test cases"""
    pass

