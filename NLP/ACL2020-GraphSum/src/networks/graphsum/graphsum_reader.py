#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Data reader for GraphSum model"""

import os
import json
import glob
import numpy as np
from collections import namedtuple
import paddle.fluid as fluid
from utils.logging import logger


class GraphSumReader(object):
    """GraphSum data reader"""

    def __init__(self, max_para_num=30, max_para_len=60, max_tgt_len=150,
                 graph_type="similarity", in_tokens=False, random_seed=None,
                 bos_idx=0, eos_idx=1, pad_idx=2, n_head=8):

        self.max_para_num = max_para_num
        self.max_para_len = max_para_len
        self.max_tgt_len = max_tgt_len
        self.graph_type = graph_type
        self.in_tokens = in_tokens
        self.n_head = n_head

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        if not random_seed:
            random_seed = 0
        np.random.seed(random_seed)

        self.trainer_id = 0
        self.trainer_nums = 1
        if os.getenv("PADDLE_TRAINER_ID"):
            self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        if os.getenv("PADDLE_NODES_NUM"):
            self.trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM"))

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = -1
        self.features = {}

    def load_dataset(self, data_path, shuffle=False):
        """
        Dataset generator. Don't do extra stuff here, like printing,
        because they will be postponed to the first loading time.
        Returns:
            A list of examples.
        """

        def _dataset_loader(pt_file):
            dataset = json.load(open(pt_file))
            logger.info('Loading dataset from %s, number of examples: %d' %
                        (pt_file, len(dataset)))
            return dataset

        # Sort the glob output by file name (by increasing indexes).
        pts = sorted(glob.glob(data_path + '/*.[0-9]*.json'))
        if pts:
            if shuffle:
                np.random.shuffle(pts)

            datasets = []
            for pt in pts:
                datasets.extend(_dataset_loader(pt))
            return datasets
        else:
            # Only one inputters.*Dataset, simple!
            pts = sorted(glob.glob(data_path + '/*.json'))
            dataset = _dataset_loader(pts[0])
            if shuffle:
                np.random.shuffle(dataset)

            return dataset

    def lazy_load_dataset(self, data_path, shuffle=False):
        """
        Dataset generator. Don't do extra stuff here, like printing,
        because they will be postponed to the first loading time.
        Returns:
            A list of examples.
        """

        def _dataset_loader(pt_file):
            dataset = json.load(open(pt_file))
            logger.info('Loading dataset from %s, number of examples: %d' %
                        (pt_file, len(dataset)))
            return dataset

        # Sort the glob output by file name (by increasing indexes).
        pts = sorted(glob.glob(data_path + '/*.[0-9]*.json'))
        if pts:
            if shuffle:
                np.random.shuffle(pts)

            for pt in pts:
                yield _dataset_loader(pt)

        else:
            # Only one inputters.*Dataset, simple!
            pts = sorted(glob.glob(data_path + '/*.json'))
            yield _dataset_loader(pts[0])

    def get_num_examples(self, data_path):
        """Get the total num of examples in dataset"""
        if self.num_examples != -1:
            return self.num_examples

        num_examples = 0
        dataset_loader = self.lazy_load_dataset(data_path)
        for dataset in dataset_loader:
            num_examples += len(dataset)
        self.num_examples = num_examples
        
        return self.num_examples

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_examples(self, data_path, shuffle=False):
        """Reads json dict file."""
        data_id = 0
        reader = self.load_dataset(data_path, shuffle)
        Example = namedtuple('Example', ["src", "tgt", "tgt_str", "graph", "data_id"])

        assert self.graph_type in ["similarity", "topic", "discourse"], "Non-valid graph type!"

        examples = []
        for ex in reader:
            if self.graph_type == "similarity":
                graph = ex['sim_graph']
            elif self.graph_type == "topic":
                graph = ex['topic_graph']
            else:
                graph = ex['discourse_graph']
            examples.append(Example(src=ex['src'], tgt=ex['tgt'], tgt_str=ex['tgt_str'],
                                    graph=graph, data_id=data_id))
            data_id += 1

        return examples

    def _example_reader(self, data_path, shuffle=False):
        """Reads json dict file."""
        data_id = 0
        reader = self.lazy_load_dataset(data_path, shuffle)
        Example = namedtuple('Example', ["src", "tgt", "tgt_str", "graph", "data_id"])

        assert self.graph_type in ["similarity", "topic", "discourse"], "Non-valid graph type!"

        for dataset in reader:
            if shuffle:
                np.random.shuffle(dataset)

            for ex in dataset:
                if self.graph_type == "similarity":
                    graph = ex['sim_graph']
                elif self.graph_type == "topic":
                    graph = ex['topic_graph']
                else:
                    graph = ex['discourse_graph']
                ex = Example(src=ex['src'], tgt=ex['tgt'], tgt_str=ex['tgt_str'],
                             graph=graph, data_id=data_id)
                data_id += 1

                yield ex

    def _convert_example_to_record(self, example):
        """Converts a single `Example` into a single `Record`."""
        tgt = example.tgt[:self.max_tgt_len][:-1] + [self.eos_idx]
        # truncate too long paragraph
        src = [sent[:self.max_para_len] for sent in example.src]
        src = src[:self.max_para_num]

        graph = example.graph[:self.max_para_num]
        graph = [sim[:self.max_para_num] for sim in graph]

        Record = namedtuple('Record', ['src_ids', 'tgt_ids', 'label_ids', 'graph', 'data_id'])
        record = Record(src, tgt[:-1], tgt[1:], graph, example.data_id)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None, do_dec=False, place=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        index = 0
        for example in examples:
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example)

            max_len = max(max_len, len(record.tgt_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records, do_dec, place)
                batch_records, max_len = [record], len(record.tgt_ids)
            index += 1

        if batch_records:
            yield self._pad_batch_records(batch_records, do_dec, place)

    def get_features(self, phase):
        """Get features for the dataset"""
        return self.features[phase]

    def data_generator(self, data_path, batch_size, epoch, dev_count=1,
                       shuffle=True, phase=None, do_dec=False, place=None):
        """get data batch"""

        examples = self._read_examples(data_path)

        if phase != "train":
            features = {}
            for example in examples:
                features[example.data_id] = example
            self.features[phase] = features

        def wrapper():
            """data wrapper"""
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                    trainer_id = self.trainer_id
                else:
                    trainer_id = 0
                    assert dev_count == 1, "only supports 1 GPU while prediction"

                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase, do_dec=do_dec, place=place):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        yield all_dev_batches[trainer_id]
                        all_dev_batches = []

        return wrapper

    def data_generator_with_buffer(self, data_path, batch_size, epoch, dev_count=1,
                                   shuffle=True, phase=None, do_dec=False, place=None):
        """get data batch"""
        if do_dec:
            return self.data_generator(data_path, batch_size, epoch, dev_count, shuffle,
                                       phase, do_dec, place)

        def wrapper():
            """data wrapper"""
            all_dev_batches = []
            for epoch_index in range(epoch):
                example_reader = self._example_reader(data_path, shuffle)

                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                    trainer_id = self.trainer_id
                else:
                    trainer_id = 0
                    assert dev_count == 1, "only supports 1 GPU while prediction"

                for batch_data in self._prepare_batch_data(
                        example_reader, batch_size, phase=phase, do_dec=do_dec, place=place):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        yield all_dev_batches[trainer_id]
                        all_dev_batches = []

        return wrapper

    def _pad_batch_records(self, batch_records, do_dec, place):
        """Pad data to batch"""

        if do_dec:
            return self._prepare_infer_input(batch_records, place=place)
        else:
            return self._prepare_train_input(batch_records)

    def _prepare_train_input(self, insts):
        """
        Put all padded data needed by training into a list.
        """
        src_word, src_word_pos, src_sent_pos, src_word_slf_attn_bias, \
        src_sents_slf_attn_bias, graph_attn_bias = self._pad_src_batch_data(
            insts=[inst.src_ids for inst in insts],
            graphs=[inst.graph for inst in insts])

        trg_word, trg_pos, trg_slf_attn_bias = self._pad_tgt_batch_data(
            insts=[inst.tgt_ids for inst in insts])

        lbl_word, lbl_weight = self._pad_label_batch_data(
            insts=[inst.label_ids for inst in insts])

        data_inputs = [
            src_word, src_word_pos, src_sent_pos, src_word_slf_attn_bias,
            src_sents_slf_attn_bias, graph_attn_bias, trg_word, trg_pos,
            trg_slf_attn_bias, lbl_word, lbl_weight
        ]

        return data_inputs

    def _prepare_infer_input(self, insts, place):
        """
        Put all padded data needed by beam search decoder into a list.
        """
        src_word, src_word_pos, src_sent_pos, src_word_slf_attn_bias, \
        src_sents_slf_attn_bias, graph_attn_bias = self._pad_src_batch_data(
            insts=[inst.src_ids for inst in insts],
            graphs=[inst.graph for inst in insts])

        # start tokens
        trg_word = np.asarray([[self.bos_idx]] * len(insts), dtype="int64")

        def to_lodtensor(data, place, lod=None):
            """convert to lod_tensor"""
            data_tensor = fluid.LoDTensor()
            data_tensor.set(data, place)
            if lod is not None:
                data_tensor.set_lod(lod)
            return data_tensor

        # beamsearch_op must use tensors with lod
        init_score = to_lodtensor(
            np.zeros_like(trg_word, dtype="float32"),
            place, [range(trg_word.shape[0] + 1)] * 2)
        trg_word = to_lodtensor(trg_word, place, [range(trg_word.shape[0] + 1)] * 2)

        init_idx = np.asarray(range(len(insts)), dtype="int32")

        batch_data_ids = np.array([inst.data_id for inst in insts], dtype="int64").reshape([-1, 1])

        data_inputs = [
            src_word, src_word_pos, src_sent_pos, src_word_slf_attn_bias,
            src_sents_slf_attn_bias, graph_attn_bias, trg_word, init_score,
            init_idx, batch_data_ids
        ]
        return data_inputs

    def _pad_matrix(self, data, height, width, pad_id):
        """ padding the input with height paragraphs and each paragraph have width words """
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        return rtn_data

    def _pad_src_batch_data(self, insts, graphs):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []

        # (batch_size, max_nblock, max_ntoken)
        inst_data = np.array([self._pad_matrix(inst, self.max_para_num, self.max_para_len, self.pad_idx)
                              for inst in insts], dtype="int64")
        return_list += [inst_data]

        # (batch_size, max_nblock, max_ntoken)
        inst_word_pos = np.array([[list(range(0, len(para))) + [0] * (self.max_para_len - len(para))
                                   for para in inst] + [[0] * self.max_para_len] * (self.max_para_num - len(inst))
                                  for inst in insts], dtype="int64")
        return_list += [inst_word_pos]

        # (batch_size, max_nblock)
        inst_sent_pos = np.array([list(range(0, len(inst))) + [0] *
                                  (self.max_para_num - len(inst)) for inst in insts], dtype="int64")
        return_list += [inst_sent_pos]

        # This is used to avoid attention on paddings.
        # (batch_size, max_nblock, max_ntoken)
        src_words_slf_attn_bias_data = np.array([[[0] * len(para) + [-1e18] * (self.max_para_len - len(para))
                                                  for para in inst] +
                                                 [[-1e18] * self.max_para_len] * (self.max_para_num - len(inst))
                                                 for inst in insts], dtype="float32")
        return_list += [src_words_slf_attn_bias_data]

        # (batch_size, max_nblock)
        src_sents_slf_attn_bias_data = np.array([[0] * len(inst) + [-1e18] * (self.max_para_num - len(inst))
                                                 for inst in insts], dtype="float32")
        return_list += [src_sents_slf_attn_bias_data]

        graphs = [[[1.0 - float(sim) for sim in list(row)] for row in g] for g in graphs]
        # (batch_size, max_nblock, max_nblock)
        graph_attn_bias = np.array([self._pad_matrix(g, self.max_para_num, self.max_para_num, 1.0)
                                    for g in graphs], dtype="float32")
        return_list += [graph_attn_bias]

        return return_list

    def _pad_tgt_batch_data(self, insts):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []

        # (batch_size, max_tgt_len)
        inst_data = np.array([inst + [self.pad_idx] * (self.max_tgt_len - len(inst)) for inst in insts],
                             dtype="int64")
        return_list += [inst_data]

        # (batch_size, max_tgt_len)
        inst_pos = np.array([list(range(0, len(inst))) + [0] *
                             (self.max_tgt_len - len(inst)) for inst in insts], dtype="int64")
        return_list += [inst_pos]

        # This is used to avoid attention on subsequent words.
        slf_attn_bias_data = np.ones((len(insts), self.max_tgt_len, self.max_tgt_len), dtype="float32")
        slf_attn_bias_data = np.triu(slf_attn_bias_data, 1) * -1e18
        return_list += [slf_attn_bias_data]

        return return_list

    def _pad_label_batch_data(self, insts):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []

        # (batch_size, max_tgt_len)
        inst_data = np.array([inst + [self.pad_idx] * (self.max_tgt_len - len(inst)) for inst in insts],
                             dtype="int64")
        return_list += [inst_data]

        # (batch_size, max_tgt_len)
        inst_weight = np.array([[1.] * len(inst) + [0.] * (self.max_tgt_len - len(inst))
                                for inst in insts], dtype="float32")
        return_list += [inst_weight]

        return return_list


if __name__ == '__main__':
    pass
