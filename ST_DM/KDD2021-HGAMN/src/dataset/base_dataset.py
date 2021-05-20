# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Base Dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import six
from io import open
from collections import namedtuple
import numpy as np
import tqdm
import paddle
from pgl.utils import mp_reader
import collections
import logging
log = logging.getLogger(__name__)

import pgl

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def batch_iter(data, batch_size, fid, num_workers):
    """node_batch_iter
    """
    size = len(data)
    perm = np.arange(size)
    np.random.shuffle(perm)
    start = 0
    cc = 0
    while start < size:
        index = perm[start:start + batch_size]
        start += batch_size
        cc += 1
        if cc % num_workers != fid:
            continue
        yield data[index]


def scan_batch_iter(data, batch_size, fid, num_workers):
    """node_batch_iter
    """
    batch = []
    cc = 0
    for line_example in data.scan(): 
        cc += 1
        if cc % num_workers != fid:
            continue
        batch.append(line_example)
        if len(batch) == batch_size:
            yield batch 
            batch = []

    if len(batch) > 0:
        yield batch 


class BaseDataGenerator(object):
    """Base Data Geneartor"""

    def __init__(self, buf_size, batch_size, num_workers, shuffle=True):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.line_examples = []
        self.buf_size = buf_size
        self.shuffle = shuffle

    def batch_fn(self, batch_examples):
        """ batch_fn batch producer"""
        raise NotImplementedError("No defined Batch Fn")

    def batch_iter(self, fid):
        """ batch iterator"""
        if self.shuffle:
            for batch in batch_iter(self, self.batch_size, fid, self.num_workers):
                yield batch
        else:
            for batch in scan_batch_iter(self, self.batch_size, fid, self.num_workers):
                yield batch

    def __len__(self):
        return len(self.line_examples)

    def __getitem__(self, idx):
        if isinstance(idx, collections.Iterable):
            return [self[bidx] for bidx in idx]
        else:
            return self.line_examples[idx]

    def generator(self):
        """batch dict generator"""

        def worker(filter_id):
            """ multiprocess worker"""

            def func_run():
                """ func_run """
                for batch_examples in self.batch_iter(filter_id):
                    batch_dict = self.batch_fn(batch_examples)
                    yield batch_dict

            return func_run

        if self.num_workers == 1:
            r = paddle.reader.buffered(worker(0), self.buf_size)
        else:
            worker_pool = [worker(wid) for wid in range(self.num_workers)]
            worker = mp_reader.multiprocess_reader(
                worker_pool, use_pipe=True, queue_size=1000)
            r = paddle.reader.buffered(worker, self.buf_size)

        for batch in r():
            yield batch

    def scan(self): 
        for line_example in self.line_examples:
            yield line_example
