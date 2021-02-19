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
 Specify the brief sample_reader.py
 Date: 2019/07/23 20:45:56
 Brief:
      Generate sample for pyreader and datafeeder
"""

from __future__ import print_function

import numpy as np
import logging
import sys
import os
import random
import six


def stdin_gen():
    """
        stdin gen wrapper
    """
    if six.PY3:
        source = sys.stdin.buffer 
    else:
        source = sys.stdin
    while True:
        line = source.readline()
        if len(line) == 0:
            break
        yield line


class SampleReader(object):
    """
    PyReader interface
    """
    
    @classmethod
    def get_sample_reader(cls, dataset_instance, input_names):
        """
        return pyreader object.
        """
        def _data_generator(): 
            #distributed predict get sample from stdin
            if dataset_instance._flags.platform == "hadoop":
                """
                stdin sample generator: read from stdin 
                """
                for line in sys.stdin:
                    if not line.strip():
                        continue
                    yield line
            else:
                """
                filelist sample generator: read from file 
                """
                logging.info("current worker file_list: %s" % dataset_instance._flags.file_list)

                for fname in dataset_instance._flags.file_list.split(','):
                    with open(fname, 'r') as fp:
                        for line in fp:
                            if not line.strip():
                                continue
                            yield line

        def _one_reader():
            for line in _data_generator():
                for sample in dataset_instance.parse_oneline(line):
                    sample_list = [value for key, value in sample if key in input_names]
                    yield sample_list

        def _batch_reader():
            for batch in dataset_instance.parse_batch(_data_generator):
                sample_list = [value for key, value in batch if key in input_names]
                yield sample_list

        if dataset_instance._flags.reader_batch:
            return _batch_reader
                        
        return _one_reader

