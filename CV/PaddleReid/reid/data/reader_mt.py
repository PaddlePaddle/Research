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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import functools
import collections
import traceback
import numpy as np
import logging
import random

import pdb

from collections import defaultdict


from . transform import build_transform


from .parallel_map import ParallelMap
__all__ = ['ReaderMT', 'create_readerMT']

logger = logging.getLogger(__name__)


class Compose(object):
    def __init__(self, transforms, ctx=None):
        self.transforms = transforms
        self.ctx = ctx

    def __call__(self, data):
        ctx = self.ctx if self.ctx else {}
        for f in self.transforms:
            try:
                data = f(data, ctx)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.info("fail to map op [{}] with error: {} and stack:\n{}".
                            format(f, e, str(stack_info)))
                raise e
        return data


def _calc_img_weights(roidbs):
    """ calculate the probabilities of each sample
    """
    imgs_cls = []
    num_per_cls = {}
    img_weights = []
    for i, roidb in enumerate(roidbs):
        img_cls = set([k for cls in roidbs[i]['gt_class'] for k in cls])
        imgs_cls.append(img_cls)
        for c in img_cls:
            if c not in num_per_cls:
                num_per_cls[c] = 1
            else:
                num_per_cls[c] += 1

    for i in range(len(roidbs)):
        weights = 0
        for c in imgs_cls[i]:
            weights += 1 / num_per_cls[c]
        img_weights.append(weights)
    # probabilities sum to 1
    img_weights = img_weights / np.sum(img_weights)
    return img_weights


def batch_arrange(batch_samples, fields):

    arrange_batch = []
    for samples in batch_samples:
        one_ins = ()
        for i, field in enumerate(fields):
            one_ins += (samples[field], )
        arrange_batch.append(one_ins)
    return arrange_batch

def StandardizeDatasetMT(dataset, root='./downsample_vehicle/images'):

    num_samples = len(dataset)
    standard = []
    for i in range(num_samples):
        # fname, pid, camid = dataset[i]

        fname, pid, camid, colorid, typeid, orientation = dataset[i]
        one_sample = {}
        one_sample['im_file'] = os.path.join(root,fname)
        one_sample['pid'] = pid
        one_sample['camid'] = camid - 1
        one_sample['index'] = i
        one_sample['colorid'] = colorid
        one_sample['typeid'] = typeid
        one_sample['orientation'] = orientation

        standard.append(one_sample)
    return standard

class RandomIdentitySampler(object):
    def __init__(self, data_source, num_batch_pids=16, num_instances=1):
        self.data_source = data_source
        self.num_batch_pids = num_batch_pids
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        self.num_images = len(data_source)

        for i in range(self.num_images):
            cur_sample = data_source[i]
            pid = cur_sample['pid']
            self.index_dic[pid].append(i)
        self.pids = list(self.index_dic.keys())
        
        self.num_classes = len(self.pids)
        num_samples = len(self.pids)
        self.index_list = np.arange(num_samples)
        self.num_last_pids = num_samples % self.num_batch_pids
        self.num_samples = num_samples - self.num_last_pids
        self.num_iters_per_epoch = int(self.num_samples / self.num_batch_pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def reset(self):
        indices = self.index_list.copy()
        np.random.shuffle(indices)
        indices = indices[:-self.num_last_pids]
        
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return ret

class BaseSampler(object):
    def __init__(self, data_source, shuffle=False):
        #pdb.set_trace()
        self.data_source = data_source
        self.num_images = len(self.data_source)
        self.index_list = np.arange(self.num_images)
        self.shuffle = shuffle

    def __len__(self):
        return self.num_images

    def reset(self):
        indices = self.index_list.copy()
        if self.shuffle:
            np.random.shuffle(indices)
        return indices



class ReaderMT(object):
    """
    Args:
        dataset (DataSet): DataSet object
        sample_transforms (list of BaseOperator): a list of sample transforms
            operators.
        batch_transforms (list of BaseOperator): a list of batch transforms
            operators.
        batch_size (int): batch size.
        shuffle (bool): whether shuffle dataset or not. Default False.
        drop_last (bool): whether drop last batch or not. Default True.
        drop_empty (bool): whether drop sample when it's gt is empty or not.
            Default True.
        mixup_epoch (int): mixup epoc number. Default is -1, meaning
            not use mixup.
        class_aware_sampling (bool): whether use class-aware sampling or not.
            Default False.
        worker_num (int): number of working threads/processes.
            Default -1, meaning not use multi-threads/multi-processes.
        use_process (bool): whether use multi-processes or not.
            It only works when worker_num > 1. Default False.
        bufsize (int): buffer size for multi-threads/multi-processes,
            please note, one instance in buffer is one batch data.
        memsize (str): size of shared memory used in result queue when
            use_process is true. Default 3G.
        inputs_def (dict): network input definition use to get input fields,
            which is used to determine the order of returned data.
    """

    def __init__(self,
                 dataset=None,
                 img_dir=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=None,
                 num_instances=None,
                 sample_type='Identity', # ['Identity', 'Base']
                 shuffle=False,
                 drop_last=True,
                 worker_num=-1,
                 use_process=False,
                 bufsize=100,
                 memsize='3G',
                 input_fields=['image','camid','colorid', 'typeid'],
                 is_test = False,
                 cfg=None):
        if not isinstance(dataset[0],dict):
            self._roidbs = StandardizeDatasetMT(dataset, img_dir)
        else:
            self._roidbs = dataset
        self._fields = input_fields

        # transform
        assert cfg!=None
        train_transform, test_transform = build_transform(cfg)


        if is_test:
            self._sample_transforms = Compose(test_transform,
                                          {'fields': self._fields})
        else:
            self._sample_transforms = Compose(train_transform,
                                          {'fields': self._fields})
        self._batch_transforms = None


        # if batch_transforms:
        #     self._batch_transforms = Compose(batch_transforms,
        #                                      {'fields': self._fields})

        # data
        self._batch_size = batch_size
        if is_test:
            self._num_instances=1
        else:
            self._num_instances = num_instances
        assert self._batch_size % self._num_instances == 0
        self._num_batch_pids = int(self._batch_size / self._num_instances)
        self._shuffle = shuffle
        self._drop_last = drop_last
        if sample_type == 'Identity':
            assert self._shuffle == True
            assert self._drop_last == True
            self.index_sampler = RandomIdentitySampler(self._roidbs, self._num_batch_pids, self._num_instances)
            self.num_classes = self.index_sampler.num_classes
            self.num_iters_per_epoch = self.index_sampler.num_iters_per_epoch
            self.num_batch_pids = self.index_sampler.num_batch_pids
        else:
            self.index_sampler = BaseSampler(self._roidbs, self._shuffle)
            self.num_classes = 1695
            self.num_iters_per_epoch = -1 
            self.num_batch_pids = -1 
        # sampling
        self._load_img = False
        self._sample_num = len(self.index_sampler)
        #pdb.set_trace()

        self._indexes = None

        self._pos = -1
        self._epoch = -1

        # multi-process
        self._worker_num = worker_num
        self._parallel = None
        if self._worker_num > -1:
            self._parallel = ParallelMap(self, self.worker, worker_num, bufsize,
                                         use_process, memsize)

    def __call__(self):
        if self._worker_num > -1:
            return self._parallel
        else:
            return self

    def __iter__(self):
        return self

    def reset(self):
        """implementation of Dataset.reset
        """
        self.indexes = self.index_sampler.reset()
        self._sample_num = len(self.indexes)

        if self._epoch < 0:
            self._epoch = 0
        else:
            self._epoch += 1

        self._pos = 0

    def __next__(self):
        return self.next()

    def next(self):
        if self._epoch < 0:
            self.reset()
        if self.drained():
            raise StopIteration
        batch = self._load_batch()
        if self._drop_last and len(batch) < self._batch_size:
            raise StopIteration
        if self._worker_num > -1:
            return batch
        else:
            return self.worker(batch)

    def _load_batch(self):
        batch = []
        bs = 0
        while bs != self._batch_size:
            if self._pos >= self.size():
                break
            pos = self.indexes[self._pos]
            sample = copy.deepcopy(self._roidbs[pos])
            #pdb.set_trace()
            self._pos += 1

            if self._load_img:
                sample['image'] = self._load_image(sample['im_file'])

            batch.append(sample)
            bs += 1
        return batch

    def worker(self, batch_samples=None):
        """
        sample transform and batch transform.
        """
        batch = []
        for sample in batch_samples:
            sample = self._sample_transforms(sample)
            batch.append(sample)
        if len(batch) > 0 and self._batch_transforms:
            batch = self._batch_transforms(batch)
        if len(batch) > 0 and self._fields:
            batch = batch_arrange(batch, self._fields)
        return batch

    def _load_image(self, filename):
        with open(filename, 'rb') as f:
            return f.read()

    def size(self):
        """ implementation of Dataset.size
        """
        return self._sample_num

    def drained(self):
        """ implementation of Dataset.drained
        """
        assert self._epoch >= 0, 'The first epoch has not begin!'
        return self._pos >= self.size()

    def stop(self):
        if self._parallel:
            self._parallel.stop()


def create_readerMT(cfg, max_iter=0):
    """
    Return iterable data reader.

    Args:
        max_iter (int): number of iterations.
    """
    if not isinstance(cfg, dict):
        raise TypeError("The config should be a dict when creating reader.")

    # synchornize use_fine_grained_loss/num_classes from global_cfg to reader cfg
    reader = ReaderMT(**cfg)()
    num_classes = reader.num_classes
    num_batch_pids = reader.num_batch_pids
    num_iters_per_epoch = reader.num_iters_per_epoch

    def _reader():
        n = 0
        while True:
            for _batch in reader:
                if len(_batch) > 0:
                    yield _batch
                    n += 1
                if max_iter > 0 and n == max_iter:
                    return
            reader.reset()
            if max_iter <= 0:
                return

    return _reader, num_classes, num_batch_pids, num_iters_per_epoch
