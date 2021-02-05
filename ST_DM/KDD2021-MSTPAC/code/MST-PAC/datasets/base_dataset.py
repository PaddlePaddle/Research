#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################


"""
 Specify the brief base_dataset.py
 Date: 2019/07/10 16:32:52
 Brief:
    Basic dataset, 
    should be inheritted by user-defined dataset class.
"""

import os
import sys
from collections import OrderedDict
import six

from datasets.datasets_factory import Meta

class RegisterBaseDataset(object):
    """
    Base dataset with abstract interface
    """
    def __init__(self, flags):
        self._flags = flags
        self.is_training = (self._flags.dataset_split_name == 'train')

    def parse_context(self, inputs):
        """
        provide context for frame
        """
        return {}

    def parse_oneline(self, line):
        """
        parse sample line with the sample corresponding format
        """
        result = {}

        return result
    
    def parse_batch(self, data_gen):
        """
        parse sample line with the sample corresponding format
        """
        result = {}

        return result

    def get_dataset_info(self):
        """
        get dataset info
        """
        #print "base_dataset get_dataset_info"
        return {}


@six.add_metaclass(Meta)
class BaseDataset(RegisterBaseDataset):
    """
    Set metclass for sub-class register
    """
    pass 


