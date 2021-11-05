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
import itertools
import numpy as np
from collections import namedtuple
import paddle.fluid as fluid
from utils.logging import logger

class GraphSumReader(object):
    """GraphSum data reader"""

    def __init__(self, max_para_num=30, max_para_len=60, max_tgt_len=150, max_doc_num=10,
                 graph_type="similarity", in_tokens=False, random_seed=None,
                 bos_idx=0, eos_idx=1, pad_idx=2, n_head=8, c_sent_num=5, s_sent_num=3):

        self.max_para_num = max_para_num
        self.max_para_len = max_para_len
        self.max_doc_num = max_doc_num
        self.max_tgt_len = max_tgt_len
        self.graph_type = graph_type
        self.in_tokens = in_tokens
        self.n_head = n_head

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.c_sent_num = c_sent_num
        self.s_sent_num = s_sent_num
        self.c_summary_num = len(list(itertools.combinations(list(range(self.c_sent_num)), self.s_sent_num)))

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
        self.labeled_sent_num = 5
        self.emb_size = 768

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
        # for predict process
        data_id = 0
        reader = self.load_dataset(data_path, shuffle)
        Example = namedtuple('Example', ["src", "tgt", "tgt_str", "graph", "data_id", "sent_labels", "src_str", "cls_ids", "sep_ids", "summary_rank"])

        assert self.graph_type in ["similarity", "topic", "discourse"], "Non-valid graph type!"

        examples = []
        for ex in reader:
            need_skip = False
            doc_num = len(ex['src'])
            if self.graph_type == "similarity":
                graph = ex['sim_graph']
            elif self.graph_type == "topic":
                graph = ex['topic_graph']
            else:
                graph = ex['discourse_graph']
            examples.append(Example(src=ex['src'], tgt=ex['tgt'], tgt_str=ex['tgt_str'],
                                    graph=graph, data_id=data_id, sent_labels=ex["sent_labels"], 
                                    src_str=ex["src_str"], cls_ids=ex["cls_ids"], sep_ids=ex["sep_ids"], summary_rank=ex["summary_rank"]))
            data_id += 1

        return examples

    def _example_reader(self, data_path, shuffle=False):
        """Reads json dict file."""
        # for train process
        data_id = 0
        reader = self.lazy_load_dataset(data_path, shuffle)
        Example = namedtuple('Example', ["src", "tgt", "tgt_str", "graph", "data_id", "sent_labels", "cls_ids", "sep_ids", "summary_rank"])

        assert self.graph_type in ["similarity", "topic", "discourse"], "Non-valid graph type!"
        count = 0
        for dataset in reader:
            if shuffle:
                np.random.shuffle(dataset)

            for ex in dataset:
                need_skip = False
                doc_num = len(ex['src'])
                if self.graph_type == "similarity":
                    graph = ex['sim_graph']
                elif self.graph_type == "topic":
                    graph = ex['topic_graph']
                else:
                    graph = ex['discourse_graph']
                ex = Example(src=ex['src'], tgt=ex['tgt'], tgt_str=ex['tgt_str'],
                             graph=graph, data_id=data_id, sent_labels=ex["sent_labels"],
                             cls_ids=ex["cls_ids"], sep_ids=ex["sep_ids"], summary_rank=ex["summary_rank"])
                data_id += 1

                yield ex

    def _convert_example_to_record(self, example, batch_size):
        """Converts a single `Example` into a single `Record`."""
        tgt = example.tgt[:self.max_tgt_len][:-1] + [self.eos_idx]
        doc_num = len(example.src)
        multi_doc_src = []
        multi_doc_graph = []
        multi_doc_sent_labels = []
        multi_doc_cls_ids = []
        multi_doc_sep_ids = []
        multi_doc_summary_rank = []
        multi_doc_src_token_num = 0
        total_label_num = 0
        total_sent_num = 0
        for i in range(doc_num):
            # truncate too long paragraph
            src = example.src[i][:self.max_para_len]
            multi_doc_src_token_num += len(src)
            if multi_doc_src_token_num > batch_size:
                continue
            sent_labels = example.sent_labels[i][:self.max_para_num]

            cls_ids = example.cls_ids[i][:self.max_para_num]
            cls_ids = [id for id in cls_ids if id < self.max_para_len]
            true_sent_num = len(cls_ids)
            sep_ids = example.sep_ids[i][:true_sent_num]
            sent_labels = sent_labels[:true_sent_num]
            total_label_num += len([label for label in sent_labels if label == 1])
            total_sent_num += len(sent_labels)

            summary_rank = example.summary_rank[i]
            for c_summary in summary_rank:
                for sent_id in c_summary:
                    if sent_id >= true_sent_num:
                        c_summary.remove(sent_id)
            
            graph = example.graph[i][:true_sent_num]
            graph = [sim[:true_sent_num] for sim in graph]

            multi_doc_src.append(src)
            multi_doc_graph.append(graph)
            multi_doc_sent_labels.append(sent_labels)
            multi_doc_cls_ids.append(cls_ids)
            multi_doc_sep_ids.append(sep_ids)
            multi_doc_summary_rank.append(summary_rank)

        if total_label_num == 0:
            return None
        if total_sent_num < self.c_sent_num:
            return None
        Record = namedtuple('Record', ['src_ids', 'tgt_ids', 'label_ids', 'graph', 'data_id', "sent_labels", "cls_ids", "sep_ids", "summary_rank"])
        record = Record(multi_doc_src, tgt[:-1], tgt[1:], multi_doc_graph, example.data_id, multi_doc_sent_labels, multi_doc_cls_ids, multi_doc_sep_ids, multi_doc_summary_rank)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None, do_dec=False, place=None):
        """generate batch records"""
        def cal_doc_item_num(multi_doc_data):
            """calculate multi-doc item number
            """
            count = 0
            for each_doc in multi_doc_data:
                count += len(each_doc)
            return count
        
        def cal_doc_max_sent_num(multi_doc_data):
            """calculate max sent number in one multi-doc data
            """
            max_sent_num = 0
            for each_doc in multi_doc_data:
                max_sent_num = max(max_sent_num, len(each_doc))
            return max_sent_num
        
        def cal_doc_max_label_num(multi_doc_data):
            """calculate max label number in one multi-doc data
            """
            max_label_num = 0
            for each_doc in multi_doc_data:
                max_label_num = max(max_label_num, len([i for i in each_doc if i == 1]))
            return max_label_num
        
        def cal_doc_total_sent_num(multi_doc_data):
            """calculate max label number in one multi-doc data
            """
            max_label_num = 0
            for each_doc in multi_doc_data:
                max_label_num += len(each_doc)
            return max_label_num

        batch_records, max_len = [], 0
        index = 0
        batch_max_sent_num = 0
        batch_max_doc_num = 0
        batch_max_label_num = 0
        for example in examples:
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, batch_size)
            if not record:
                continue

            label_num = cal_doc_max_label_num(record.sent_labels)
            sent_num = cal_doc_max_sent_num(record.sent_labels)
            max_len = max(max_len, cal_doc_item_num(record.src_ids))

            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_max_sent_num = max(batch_max_sent_num, sent_num)
                batch_max_doc_num = max(batch_max_doc_num, len(record.src_ids))
                batch_max_label_num = max(batch_max_label_num, label_num)
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records, do_dec, place, batch_max_sent_num, batch_max_doc_num, batch_max_label_num)
                batch_records, max_len = [record], cal_doc_item_num(record.src_ids)
                batch_max_sent_num = sent_num
                batch_max_doc_num = len(record.src_ids)
                batch_max_label_num = label_num
            index += 1

        if batch_records:
            yield self._pad_batch_records(batch_records, do_dec, place, batch_max_sent_num, batch_max_doc_num, batch_max_label_num)

    def get_features(self, phase):
        """Get features for the dataset"""
        return self.features[phase]

    def data_generator(self, data_path, batch_size, epoch, dev_count=1,
                       shuffle=True, phase=None, do_dec=False, place=None):
        """get data batch"""
        # for predict process

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

    def _pad_batch_records(self, batch_records, do_dec, place, batch_max_sent_num, batch_max_doc_num, batch_max_label_num):
        """Pad data to batch"""

        if do_dec:
            return self._prepare_infer_input(batch_records, place=place, batch_max_sent_num=batch_max_sent_num, batch_max_doc_num=batch_max_doc_num)
        else:
            return self._prepare_train_input(batch_records, 
                            batch_max_sent_num=batch_max_sent_num, batch_max_doc_num=batch_max_doc_num, batch_max_label_num=batch_max_label_num)

    def _prepare_train_input(self, insts, batch_max_sent_num, batch_max_doc_num, batch_max_label_num):
        """
        Put all padded data needed by training into a list.
        """
        src_word, src_word_pos, src_sent_pos,\
        src_word_slf_attn_bias, src_sents_slf_attn_bias, graph_attn_bias, cls_ids, sep_ids = self._pad_src_batch_data(
            insts=[inst.src_ids for inst in insts],
            graphs=[inst.graph for inst in insts],
		    cls_ids=[inst.cls_ids for inst in insts],
            sep_ids=[inst.sep_ids for inst in insts], 
            batch_max_sent_num=batch_max_sent_num,
            batch_max_doc_num=batch_max_doc_num)

        sent_labels_weight, sent_labels = self._pad_sent_label_batch_data(
            insts=[inst.sent_labels for inst in insts],
            batch_max_sent_num=batch_max_sent_num,
            batch_max_doc_num=batch_max_doc_num
        )

        summary_rank, summary_rank_high, summary_rank_low = self._pad_summary_rank_batch_data(
            insts=[inst.summary_rank for inst in insts],
            batch_max_sent_num=batch_max_sent_num,
            batch_max_doc_num=batch_max_doc_num
        )

        labeled_sent, labeled_sent_weight = self._pad_labeled_sent_batch_data(
            insts=[inst.sent_labels for inst in insts],
            batch_max_doc_num=batch_max_doc_num,
            batch_max_sent_num=batch_max_sent_num
        )

        cand_summary_combinations = self._generate_candi_summary_combinations(
            insts=[inst.cls_ids for inst in insts], 
            candi_sent_num=self.c_sent_num, selected_sent_num=self.s_sent_num, 
            batch_labeled_sent_num=batch_max_sent_num
        )

        data_inputs = [
            src_word, src_word_pos, src_sent_pos, src_word_slf_attn_bias,
            src_sents_slf_attn_bias, graph_attn_bias, sent_labels, sent_labels_weight, cls_ids, sep_ids,
            cand_summary_combinations,
            labeled_sent, labeled_sent_weight,
            summary_rank, summary_rank_high, summary_rank_low,
            # pos_sent_combine, neg_sent_combine
        ]

        return data_inputs

    def _prepare_infer_input(self, insts, place, batch_max_sent_num, batch_max_doc_num):
        """
        Put all padded data needed by beam search decoder into a list.
        """
        src_word, src_word_pos, src_sent_pos, \
        src_word_slf_attn_bias, src_sents_slf_attn_bias, graph_attn_bias, cls_ids, sep_ids = self._pad_src_batch_data(
            insts=[inst.src_ids for inst in insts],
            graphs=[inst.graph for inst in insts],
			cls_ids=[inst.cls_ids for inst in insts],
            sep_ids=[inst.sep_ids for inst in insts],
            batch_max_sent_num=batch_max_sent_num,
            batch_max_doc_num=batch_max_doc_num
        )

        batch_data_ids = np.array([inst.data_id for inst in insts], dtype="int64").reshape([-1, 1])

        sent_labels_weight, sent_labels = self._pad_sent_label_batch_data(
            insts=[inst.sent_labels for inst in insts],
            batch_max_sent_num=batch_max_sent_num,
            batch_max_doc_num=batch_max_doc_num
        )

        cand_summary_combinations = self._generate_candi_summary_combinations(
            insts=[inst.cls_ids for inst in insts], 
            candi_sent_num=self.c_sent_num, selected_sent_num=self.s_sent_num, 
            batch_labeled_sent_num=batch_max_sent_num
        )

        data_inputs = [
            src_word, src_word_pos, src_sent_pos, src_word_slf_attn_bias,
            src_sents_slf_attn_bias, graph_attn_bias, sent_labels, sent_labels_weight, 
            batch_data_ids, cls_ids, sep_ids,
            cand_summary_combinations
        ]
        return data_inputs

    def _pad_word_matrix(self, data, width, pad_id):
        """ padding the input with height paragraphs and each paragraph have width words """
        rtn_data = data + [pad_id] * (width - len(data))
        return rtn_data
    
    def _pad_sent_matrix(self, data, height, width, pad_id):
        """ padding the input with height sents and each sents have width words """
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        return rtn_data
    
    def _pad_doc_matrix(self, data, length, height, width, pad_id):
        """ padding the input with length doc numbers height sents and each sents have width words """
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        rtn_data = [[d + [pad_id] * (width - len(d)) for d in doc_data] + [[pad_id] * width] * (height - len(doc_data)) for doc_data in data]
        rtn_data = rtn_data + [[[pad_id] * width] * height] * (length - len(data))

        return rtn_data

    def _pad_src_batch_data(self, insts, graphs, cls_ids, sep_ids, batch_max_sent_num, batch_max_doc_num):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        def threshold_sim(sim):
            if sim > 0.2:
                return 0.0
            return -1e18

        return_list = []

        # (batch_size, doc_num, max_ntoken)
        inst_data = np.array([self._pad_sent_matrix(inst, batch_max_doc_num, self.max_para_len, self.pad_idx)
                              for inst in insts], dtype="int64")
        inst_data = inst_data.reshape(-1, batch_max_doc_num, self.max_para_len, 1)
        # print("inst_data: " + str(inst_data) + " shape: " + str(inst_data.shape))
        return_list += [inst_data]

	    # (batch_size, doc_num, max_ntoken)
        inst_word_pos = np.array([[list(range(2, len(para)+2)) + [self.max_para_len-1] * (self.max_para_len - len(para))
                                   for para in inst] + [[self.max_para_len-1] * self.max_para_len] * (batch_max_doc_num - len(inst))
                                  for inst in insts], dtype="int64")

        inst_word_pos = inst_word_pos.reshape(-1, batch_max_doc_num, self.max_para_len, 1)
        # print("word pos: " + str(inst_word_pos) + " shape: " + str(inst_word_pos.shape))
        return_list += [inst_word_pos]

        # (batch_size, doc_num, max_nblock)
        inst_sent_pos = np.array([[list(range(0, len(each_cls_ids))) + [self.max_para_num-1] * (batch_max_sent_num - len(each_cls_ids)) 
                                    for each_cls_ids in doc_cls_ids] + [[self.max_para_num-1] * batch_max_sent_num] * (batch_max_doc_num - len(doc_cls_ids)) 
                                    for doc_cls_ids in cls_ids], dtype="int64")
        inst_sent_pos = inst_sent_pos.reshape(-1, batch_max_doc_num, batch_max_sent_num, 1)
        # print("sent pos: " + str(inst_sent_pos) + " shape: " + str(inst_sent_pos.shape))
        return_list += [inst_sent_pos]

        # This is used to avoid attention on paddings.
        # (batch_size, doc_num, max_ntoken)
        # src_words_slf_attn_bias_data = np.array([[0] * len(inst) + [-1e18] * (self.max_para_len - len(inst)) for inst in insts], dtype="float32")
        src_words_slf_attn_bias_data = np.array([[[0] * len(para) + [-1e18] * (self.max_para_len - len(para))
                                                  for para in inst] +
                                                 [[-1e18] * self.max_para_len] * (batch_max_doc_num - len(inst))
                                                 for inst in insts], dtype="float32")
        # print("word slf attn bias: " + str(src_words_slf_attn_bias_data) + " shape: " + str(src_words_slf_attn_bias_data.shape))
        return_list += [src_words_slf_attn_bias_data]

        # (batch_size, doc_num*max_nblock, doc_num*max_nblock)
        # src_sents_slf_attn_bias_data = np.array([[0] * len(each_cls_ids) + [-1e18] * (batch_max_sent_num - len(each_cls_ids)) 
        #                                           for each_cls_ids in cls_ids], dtype="float32")
        src_sents_slf_attn_bias_data = np.array([[[0] * len(each_cls_ids) + [-1e18] * (batch_max_sent_num - len(each_cls_ids)) 
                                                for each_cls_ids in doc_cls_ids] + [[-1e18] * batch_max_sent_num] * (batch_max_doc_num - len(doc_cls_ids)) 
                                                for doc_cls_ids in cls_ids], dtype="float32")
        src_sents_slf_attn_bias_data = src_sents_slf_attn_bias_data.reshape(-1, batch_max_doc_num*batch_max_sent_num)
        src_sents_slf_attn_bias_data = np.expand_dims(src_sents_slf_attn_bias_data, axis=2)
        src_sents_slf_attn_bias_data = np.tile(src_sents_slf_attn_bias_data, (1, 1, batch_max_doc_num*batch_max_sent_num))
        # print("after: " + str(src_sents_slf_attn_bias_data) + " shape: " + str(src_sents_slf_attn_bias_data.shape))
        return_list += [src_sents_slf_attn_bias_data]

        graph = [[[[1.0 - float(sim) for sim in list(row)] for row in g] for g in doc_graphs] for doc_graphs in graphs]
        # graphs = [[[[threshold_sim(float(sim)) for sim in list(row)] for row in g] for g in doc_graphs] for doc_graphs in graphs]
        # (batch_size, doc_num, max_nblock, doc_num*max_nblock)
        graph_attn_bias = np.array([self._pad_doc_matrix(g, batch_max_doc_num, batch_max_sent_num, batch_max_doc_num*batch_max_sent_num, 1.0)
                                    for g in graphs], dtype="float32")
        return_list += [graph_attn_bias]

        # (batch_size, doc_num, max_para_num, 3)
        # print("begin")
        batch_cls_ids = []
        for batch_id, each_batch_cls_ids in enumerate(cls_ids):
            cur_batch_cls_ids = []
            for doc_id, doc_cls_ids in enumerate(each_batch_cls_ids):
                cur_doc_cls_ids = []
                for id in doc_cls_ids:
                    if id < self.max_para_len:
                        cur_doc_cls_ids.append([batch_id, doc_id, id])
                # pad data according sent num
                for i in range(batch_max_sent_num - len(cur_doc_cls_ids)):
                    cur_doc_cls_ids.append([batch_id, doc_id, self.max_para_len-1])
                cur_batch_cls_ids.append(cur_doc_cls_ids)
            # pad data according doc num
            for i in range(batch_max_doc_num - len(cur_batch_cls_ids)):
                cur_batch_cls_ids.append([[batch_id, batch_max_doc_num-1, self.max_para_len-1]] * batch_max_sent_num)
            batch_cls_ids.append(cur_batch_cls_ids)
        cls_ids = np.array(batch_cls_ids, dtype="int64")
        # print("cls_ids: " + str(cls_ids) + " shape: " + str(cls_ids.shape))
        return_list += [cls_ids]

        # (batch_size, doc_num, max_para_num, 3)
        batch_sep_ids = []
        for batch_id, each_batch_sep_ids in enumerate(sep_ids):
            cur_batch_sep_ids = []
            for doc_id, doc_sep_ids in enumerate(each_batch_sep_ids):
                cur_doc_sep_ids = []
                for id in doc_sep_ids:
                    if id < self.max_para_len:
                        cur_doc_sep_ids.append([batch_id, doc_id, id])
                for i in range(batch_max_sent_num - len(cur_doc_sep_ids)):
                    cur_doc_sep_ids.append([batch_id, doc_id, self.max_para_len-1])
                cur_batch_sep_ids.append(cur_doc_sep_ids)
            for i in range(batch_max_doc_num - len(cur_batch_sep_ids)):
                pad_doc_sep_ids = [batch_id, batch_max_doc_num-1, self.max_para_len-1]
                cur_batch_sep_ids.append([pad_doc_sep_ids] * batch_max_sent_num)
            batch_sep_ids.append(cur_batch_sep_ids)
        sep_ids = np.array(batch_sep_ids, dtype="int64")
        # print("sep_ids: " + str(sep_ids) + " shape: " + str(sep_ids.shape))
        return_list += [sep_ids]

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
    
    def _pad_sent_label_batch_data(self, insts, batch_max_sent_num, batch_max_doc_num):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding sent labels.
        """
        return_list = []

        # (batch_size, doc_num, max_nblock)
        inst_sent_labels_weight = np.array([[[1.] * len(sent_labels) + [0.] * (batch_max_sent_num - len(sent_labels))
                                for sent_labels in doc_sent_labels] + [[0.] * batch_max_sent_num] * (batch_max_doc_num - len(doc_sent_labels))
                                for doc_sent_labels in insts], dtype="float32")
        inst_sent_labels_weight = inst_sent_labels_weight.reshape(-1, batch_max_doc_num, batch_max_sent_num, 1)
        # print("sent labales weight: " + str(inst_sent_labels_weight) + " shape: " + str(inst_sent_labels_weight.shape))
        return_list += [inst_sent_labels_weight]

        # (batch_size, doc_num, max_nblock)
        inst_sent_labels = np.array([[sent_labels + [0.] * (batch_max_sent_num - len(sent_labels)) 
                                for sent_labels in doc_sent_labels] + [[0.] * batch_max_sent_num] * (batch_max_doc_num - len(doc_sent_labels))
                                for doc_sent_labels in insts], dtype="float32")
        inst_sent_labels = inst_sent_labels.reshape(-1, batch_max_doc_num, batch_max_sent_num, 1)
        # print("sent labels: " + str(inst_sent_labels) + " shape: " + str(inst_sent_labels.shape))
        return_list += [inst_sent_labels]

        return return_list
    
    def _pad_summary_rank_batch_data(self, insts, batch_max_sent_num, batch_max_doc_num):
        """ 
        Pad summary rank data
        """
        
        return_list = []
        # (batch_size, doc_num, C(3,5), 3, 3)
        inst_summary_rank = []
        for batch_id, each_batch_summary_rank in enumerate(insts):
            cur_batch_summary_rank = []
            for doc_id, doc_summary_rank in enumerate(each_batch_summary_rank):
                cur_doc_summary_rank = []
                for i in range(10 - len(doc_summary_rank)):
                    doc_summary_rank.append([batch_max_sent_num-1] * 3)
                for c_summary in doc_summary_rank:
                    c_summary_rank = []
                    for sent_id in c_summary:
                        c_summary_rank.append([batch_id, doc_id, sent_id])
                    for i in range(3 - len(c_summary_rank)):
                        c_summary_rank.append([batch_id, doc_id, batch_max_sent_num-1])
                    cur_doc_summary_rank.append(c_summary_rank)
                cur_batch_summary_rank.append(cur_doc_summary_rank)
            for i in range(batch_max_doc_num - len(cur_batch_summary_rank)):
                pad_summary_rank = []
                for j in range(3):
                    pad_summary_rank.append([batch_id, batch_max_doc_num-1, batch_max_sent_num-1])
                cur_doc_summary_rank = []
                for k in range(10):
                    cur_doc_summary_rank.append(pad_summary_rank)
                cur_batch_summary_rank.append(cur_doc_summary_rank)
            inst_summary_rank.append(cur_batch_summary_rank)
        inst_summary_rank = np.array(inst_summary_rank, dtype="int64")
        # print("summary rank: " + str(inst_summary_rank) + " shape: " + str(inst_summary_rank.shape))
        return_list += [inst_summary_rank]

        inst_summary_rank_high = inst_summary_rank[:,:,:-1,:,:]
        inst_summary_rank_high = np.array(inst_summary_rank_high, dtype="int64")
        # print("summary rank high: " + str(inst_summary_rank_high) + " shape: " + str(inst_summary_rank_high.shape))
        return_list += [inst_summary_rank_high]

        inst_summary_rank_low = inst_summary_rank[:,:,1:,:,:]
        inst_summary_rank_low = np.array(inst_summary_rank_low, dtype="int64")
        # print("summary rank low: " + str(inst_summary_rank_low) + " shape: " + str(inst_summary_rank_low.shape))
        return_list += [inst_summary_rank_low]

        return return_list
    
    def _pad_labeled_sent_batch_data(self, insts, batch_max_doc_num, batch_max_sent_num):
        """
        Pad gold summary sent ids
        """
        return_list = []

        # (batch_size, doc_num, batch_max_sent_num, 3)
        batch_label_ids = []
        for batch_id, each_batch_labels in enumerate(insts):
            cur_batch_label_ids = []
            for doc_id, doc_label_ids in enumerate(each_batch_labels):
                cur_doc_label_ids = []
                for id, label in enumerate(doc_label_ids):
                    if label == 1:
                        cur_doc_label_ids.append([batch_id, doc_id, id])
                # pad data according batch_max_sent_num
                for i in range(batch_max_sent_num - len(cur_doc_label_ids)):
                    cur_doc_label_ids.append([batch_id, doc_id, 0])
                cur_batch_label_ids.append(cur_doc_label_ids)
            # pad data according doc num
            for i in range(batch_max_doc_num - len(cur_batch_label_ids)):
                cur_batch_label_ids.append([[batch_id, batch_max_doc_num-1, 0]] * batch_max_sent_num)
            batch_label_ids.append(cur_batch_label_ids)
        batch_label_ids = np.array(batch_label_ids, dtype="int64")
        # print("batch_label_ids: " + str(batch_label_ids) + " shape: " + str(batch_label_ids.shape))
        return_list += [batch_label_ids]

        # (batch_size, doc_num, batch_max_sent_num)
        batch_label_weight = np.array([[[1.] * len([i for i in sent_labels if i == 1]) + [0.] * (batch_max_sent_num - len([i for i in sent_labels if i == 1]))
                                for sent_labels in doc_sent_labels] + [[0.] * batch_max_sent_num] * (batch_max_doc_num - len(doc_sent_labels))
                                for doc_sent_labels in insts], dtype="float32")
        # batch_label_weight = batch_label_weight.reshape(-1, batch_max_doc_num, batch_max_sent_num, 1)

        return_list += [batch_label_weight]
        return return_list
    
    def _generate_candi_summary_combinations(self, insts, candi_sent_num, selected_sent_num, batch_labeled_sent_num):
        """
        generate candidate summary sentence ids combinations
        """
        return_list = []
        # (batch_size, C(candidate_sent_num, selected_sent_num), 3, 2)
        candi_sent = list(range(candi_sent_num))
        pre_combinations = list(itertools.combinations(candi_sent, selected_sent_num))
        inst_combinations = []
        for batch_id, inst in enumerate(insts):
            cur_batch_combinations = []
            for each_combin in pre_combinations:
                cur_combin = []
                for sent_id in each_combin:
                    cur_combin.append([batch_id, sent_id])
                cur_batch_combinations.append(cur_combin)
            inst_combinations.append(cur_batch_combinations)
        inst_combinations = np.array(inst_combinations, dtype="int64")
        # print("inst combinations: " + str(inst_combinations) + " shape: " + str(inst_combinations.shape))
        # return_list += []
        return inst_combinations


if __name__ == '__main__':
    pass

