#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief sample_reader.py
 Author: map(wushilei@baidu.com)
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
import codecs


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
    def get_sample_reader(cls, dataset_instance, input_names, encoding_str='utf-8'):
        """
        return pyreader object.
        """
        def _data_generator(): 
            tasks_data = {}
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
                    with codecs.open(fname, 'r', encoding=encoding_str) as fp:
                        for line in fp:
                            if not line.strip():
                                continue
                            geohash = ''.join(line.strip("\t\n").split("\t")[2].split(','))[:20]
                            # print(geohash)
                            if geohash in tasks_data:
                                tasks_data[geohash].append(line)
                            else:
                                tasks_data[geohash] = [line]
                print("region: ", len(tasks_data))
                for gh, task in tasks_data.items():
                    for data in task:
                        yield (gh, data)
                
        def _one_reader():
            for line in _data_generator():
                idx = 0
                if not dataset_instance.is_training and dataset_instance._flags.idx_index is not None:
                    cols = line.strip("\t\n").split("\t")
                    index = int(dataset_instance._flags.idx_index)
                    if len(cols) > index and index >= 0:
                        idx = cols[index]
                    dataset_instance.idx_value = idx
                for sample in dataset_instance.parse_oneline(line):
                    sample_list = [value for key, value in sample if key in input_names]
                    yield sample_list

        def _batch_reader():
            for task in dataset_instance.parse_batch(_data_generator()):
                # print(len(task))
                task_dat = []
                for batch in task:
                    # print(batch)
                    # sample_list = [value for key, value in batch if key in input_names]
                    # print(sample_list)
                    batch_dic = {}
                    for key, value in batch:
                        if key in input_names:
                            batch_dic[key] = value
                    task_dat.append([batch_dic])
                # print(task_dat[0])
                yield task_dat

        if dataset_instance._flags.reader_batch:
            return _batch_reader
                        
        return _one_reader
