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
File: frame/core/dataset_reader.py
Date: 2019/05/07 18:56:25
Brief:
    Generate sample for dataset trainer
"""

from __future__ import print_function

import logging
import numpy as np
import logging
import sys
import os
import random
import paddle.fluid.incubate.data_generator as dg

from utils.object_transform import ObjectTransform


class DatasetReader(dg.MultiSlotDataGenerator):
    """
    data reader inherated from paddle class MultiSlotDataGenerator
    """
    def init_reader(self, obj_str, input_names):
        """
        init reader.
        """
        self.dataset_instance = ObjectTransform.pickle_loads_from_str(obj_str)
        self.input_names = ObjectTransform.pickle_loads_from_str(input_names)

    def generate_sample(self, line):
        """
        generate_sample from text line.
        """
        def _iterator():
            """
            closure 
            """
            for sample in self.dataset_instance.parse_oneline(line):
                truncate_sample = [(key, value) for key, value in sample if key in self.input_names]
                yield truncate_sample
            
        return _iterator

    def generate_batch(self, samples):
        """
        This function needs to be overridden by the user to process the
        generated samples from generate_sample(self, str) function
        It is usually used as batch processing when a user wants to
        do preprocessing on a batch of samples, e.g. padding according to
        the max length of a sample in the batch

        Args:
            samples(list tuple): generated sample from generate_sample

        Returns:
            a python generator, the same format as return value of generate_sample

        Example:

            .. code-block:: python
                import paddle.fluid.incubate.data_generator as dg
                class MyData(dg.DataGenerator):

                    def generate_sample(self, line):
                        def local_iter():
                            int_words = [int(x) for x in line.split()]
                            yield ("words", int_words)
                        return local_iter

                    def generate_batch(self, samples):
                        def local_iter():
                            for s in samples:
                                yield ("words", s[1].extend([s[1][0]]))
                mydata = MyData()
                mydata.set_batch(128)
        """
        def _local_iter():
            for sample in samples:
                yield sample

        return _local_iter


if __name__ == "__main__":
    dataset_reader = DatasetReader() 
    dataset_reader.init_reader(sys.argv[1], sys.argv[2])
    dataset_reader.run_from_stdin()

