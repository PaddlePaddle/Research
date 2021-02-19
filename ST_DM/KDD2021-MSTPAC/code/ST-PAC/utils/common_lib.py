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
 Specify the brief common_lib.py
 Date: 2019/07/24 16:46:33
"""
import os
import numpy as np
import re
import itertools
import pickle
import six
import struct
import gzip

if six.PY2:
    import operator
    def accumulate(iterable, func, initial=None):
        """
        Return running totals
        # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
        # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        """
        it = iter(iterable)
        total = initial
        if initial is None:
            try:
                total = next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total = func(total, element)
            yield total
else:
    from itertools import accumulate


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class CommonLib(object):
    """
    Common lib functions
    """

    @classmethod
    def unpack_tensor(cls, result):
        """
        decode tensor to numpy array
        """
        for i, value_i in enumerate(result):
            value_i = np.array(value_i).T.flatten().tolist()
            #value_i = np.array(value_i).T.tolist()
            unpack_str = ''
            for j, value_j in enumerate(value_i):
                unpack_str = unpack_str + str(value_j) + ' '
            result[i] = unpack_str.strip()

        return result


def find_lcs(s1, s2):
    """
        find longest common string
    """
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax: p], mmax

    
def read_gz_record(filename, parse_func):
    """
        open gz
    """
    def _gen():
        with gzip.open(filename, 'rb') as f:
            while True:
                data = f.read(struct.calcsize('i'))
                if not len(data):
                    raise StopIteration
                l, = struct.unpack('i', data)
                data = f.read(l)
                yield parse_func(data)

    return _gen


def make_gz_record(from_file, to_file, serialize_func):
    """
        make gz
    """
    try:
        with open(from_file, 'rb') as fin, gzip.open(to_file, 'wb') as fout:
            for i, line in enumerate(fin):
                line = line.strip(b'\n')
                #if i % 10000 == 0:
                #    log.debug('making gz %s => %s [%d]' % (from_file, to_file, i))
                serialized = serialize_func(line) 
                l = len(serialized)
                data = struct.pack('i%ds' % l, l, serialized)
                fout.write(data)
    except Exception as e:
        raise e
