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
"""pointwise dataset"""
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
from dataset.base_dataset import BaseDataGenerator
import collections
import logging
import tokenization
from batching import pad_batch_data_maxlen
log = logging.getLogger(__name__)

import pgl
import geohash


def parse_poi(poi):
    """parse poi"""
    poi = poi.split('\x02')
    pid, city_id = poi[0].split('_')

    city_id = int(city_id)
    name = poi[1]
    addr = poi[2] if len(poi) >= 4 else ""
    poi_geo = geohash.get_loc_gh(poi[3])
    return pid, city_id, name, addr, poi_geo


class LineExamples(object):
    """Iterator"""
    def __init__(self, data_paths):
        self.data_paths = data_paths
        
    def __iter__(self):
        return iter(self.parse_poi_data())

    def parse_poi_data(self):
        """parse data from files"""
        Example = namedtuple('Example', ['qid', 'query', "city", 'poi', 'label'])

        l = 0
        count = 0
        for path in self.data_paths:
            with open(path, mode='r', encoding="utf-8", errors='ignore') as f:
                for line in tqdm.tqdm(f):
                    try:
                        line = line.strip('\r\n').split('\t')
                        qid = line[0].split("_")[-1]
                        city = line[0].split("_")[2]
                        query = line[2]
                  
                        pos_poi = line[3].split("\x01")
                        neg_poi = line[4].split("\x01")
                        for poi in pos_poi:
                            if len(poi.split("\x02")) != 6:
                                continue
                            ex = Example(
                            qid=qid, city=int(city), query=query, poi=poi, label=1)
                            yield ex


                        for poi in neg_poi:
                            if len(poi.split("\x02")) != 6:
                                continue
                            ex = Example(
                                qid=qid, city=int(city), query=query, poi=poi, label=0)
                            yield ex
                    except Exception as e:
                        continue
    

def pointwise_parse_from_files(data_paths):
    examples = []
    if True:
        """parse data from files"""
        Example = namedtuple('Example', ['qid', 'query', "city", 'poi', 'label'])

        l = 0
        count = 0
        count_pos = 0
        count_neg = 0
        for path in data_paths:
            with open(path, mode='r', encoding="utf-8", errors='ignore') as f:
                for line in tqdm.tqdm(f):
                    try:
                        line = line.strip('\r\n').split('\t')
                        if len(line) != 5:
                            continue
                        qid = line[0].split("_")[-1]
                        city = line[0].split("_")[2]
                        query = line[1]
                        pos_poi = line[2].split("\x01")
                        neg_poi = line[3].split("\x01")
                        pos_poi = list(set(pos_poi))
                        neg_poi = list(set(neg_poi) - set(pos_poi))
                        for poi in pos_poi:
                            if len(poi.split("\x02")) != 6:
                                continue
                            count_pos += 1

                            ex = Example(
                            qid=qid, city=int(city), query=query, poi=poi, label=1)
                            examples.append(ex)


                        for poi in neg_poi:
                            if len(poi.split("\x02")) != 6:
                                continue
                            count_neg += 1
                            ex = Example(
                                qid=qid, city=int(city), query=query, poi=poi, label=0)
                            examples.append(ex)
                    except Exception as e:
                        continue
    return examples

def generate_tokens(text, tokenizer):
    """generate ernie tokens"""
    text = tokenization.convert_to_unicode(text)
    tokens_a = tokenizer.tokenize(text)

    tokens = []
    text_type_ids = []
    tokens.append("[CLS]")
    text_type_ids.append(0)
    text_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        text_type_ids.append(0)
    tokens.append("[SEP]")

    text_type_ids.append(0)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    position_ids = list(range(len(token_ids)))

    Record = namedtuple('Record',
                        ['token_ids', 'text_type_ids', 'position_ids'])
    record = Record(
        token_ids=token_ids,
        text_type_ids=text_type_ids,
        position_ids=position_ids)
    return record


class CityVocab:
    """Load Vocab"""

    def __init__(self, path):
        log.info("Load Vocab from %s" % (path))
        self.vocab = {}
        with open(path, 'r', encoding="utf-8", errors='ignore') as f:
            cc = 0
            for line in f:
                line = line.strip()
                self.vocab[line] = cc + 1
                cc += 1

    def __getitem__(self, key):
        key = str(key)
        if key not in self.vocab:
            return 0
        else:
            return self.vocab[key]


class DataGenerator(BaseDataGenerator):
    def __init__(self,
                 data_paths,
                 vocab_path,
                 graph_wrapper,
                 city_vocab="cities.vocab",
                 max_seq_len=20,
                 buf_size=1000,
                 num_workers=1,
                 batch_size=128,
                 shuffle=True,
                 token_mode="ernie",
                 is_predict=False):

        super(DataGenerator, self).__init__(
            buf_size=buf_size, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)

        self.graph_wrapper = graph_wrapper
        if shuffle:
            self.line_examples = pointwise_parse_from_files(data_paths)
        else:
            self.line_examples = LineExamples(data_paths)
        self.tokenizer = tokenization.SplitTokenizer(
            vocab_file=vocab_path, do_lower_case=True, mode=token_mode)
        self.max_seq_len = max_seq_len
        self.pad_id = self.tokenizer.vocab["[PAD]"]
        self.is_predict = is_predict
        self.city_vocab = CityVocab(city_vocab)

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [
            record.text_type_ids for record in batch_records
        ]
        batch_position_ids = [record.position_ids for record in batch_records]

        padded_token_ids, input_mask = pad_batch_data_maxlen(
            batch_token_ids,
            max_len=self.max_seq_len,
            pad_idx=self.pad_id,
            return_input_mask=True)
        padded_text_type_ids = pad_batch_data_maxlen(
            batch_text_type_ids, max_len=self.max_seq_len, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data_maxlen(
            batch_position_ids, max_len=self.max_seq_len, pad_idx=self.pad_id)

        return_list = [
            padded_token_ids.squeeze(-1), padded_text_type_ids.squeeze(-1),
            padded_position_ids.squeeze(-1), input_mask.squeeze(-1)
        ]
        return return_list

    def batch_fn(self, batch_examples):
        # batch preprocess 
        batch_texts = []
        batch_query_index = []
        batch_query_city = []
        batch_poi_index = []
        batch_city_id = []
        batch_addrs = []
        cc = 0

        edges = []
        for ex in batch_examples:
            query = ex.query
            tokens = generate_tokens(query, self.tokenizer)
            batch_texts.append(tokens)
            batch_query_index.append(cc)
            edges.append((cc, cc))
            batch_query_city.append(self.city_vocab[ex.city])
            cc += 1

        for n, ex in enumerate(batch_examples):
            poi = ex.poi
            pid, city_id, text, addr = parse_poi(poi)
            # city id
            batch_city_id.append(self.city_vocab[city_id])

            tokens = generate_tokens(text, self.tokenizer)
            addr_tokens = generate_tokens(addr, self.tokenizer)
            batch_addrs.append(addr_tokens)
            batch_texts.append(tokens)
            batch_poi_index.append(cc)
            edges.append((cc, cc))
            poi_cc = cc
            cc += 1

        batch = self._pad_batch_records(batch_texts)
        node_feat_dict = {}
        node_feat_dict["src_ids"] = batch[0]
        node_feat_dict["sent_ids"] = batch[1]
        node_feat_dict["pos_ids"] = batch[2]
        node_feat_dict["input_mask"] = batch[3]

        # Graph Modeling
        g = pgl.graph.Graph(
            num_nodes=len(batch_texts), edges=edges, node_feat=node_feat_dict)
        feed_dict = self.graph_wrapper.to_feed(g)
        batch_labels = [ex.label for ex in batch_examples]
        feed_dict["labels"] = np.array(batch_labels, dtype="float32")
        feed_dict["query_index"] = np.array(batch_query_index, dtype="int64")
        feed_dict["query_city"] = np.array(batch_query_city, dtype="int64")
        feed_dict["poi_index"] = np.array(batch_poi_index, dtype="int64")
        feed_dict["city_id"] = np.array(batch_city_id, dtype="int64")

        if self.is_predict:
            batch_query = [ex.query for ex in batch_examples]
            batch_poi = [ex.poi for ex in batch_examples]
            feed_dict["batch_query"] = np.array(batch_query)
            feed_dict["batch_poi"] = np.array(batch_poi)

        batch = self._pad_batch_records(batch_addrs)
        feed_dict["addr_src_ids"] = batch[0]
        feed_dict["addr_sent_ids"] = batch[1]
        feed_dict["addr_pos_ids"] = batch[2]
        feed_dict["addr_input_mask"] = batch[3]

        return feed_dict


def parse_poi_neigh(graph_path):
    """load poi graph"""
    d = {}
    for graph_type, path in enumerate(graph_path):
        with open(path, 'r', encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                line = line.strip().split('\t')
                pid = line[0]
                if pid not in d:
                    d[pid] = []

                for q in line[1:]:
                    d[pid].append((graph_type, q))
    return d


class GraphDataGenerator(DataGenerator):
    def __init__(self,
                 graph_path,
                 data_paths,
                 vocab_path,
                 graph_wrapper,
                 max_seq_len=20,
                 buf_size=1000,
                 num_workers=1,
                 avoid_leak=True,
                 batch_size=128,
                 shuffle=True,
                 token_mode="ernie",
                 is_predict=False):

        super(GraphDataGenerator, self).__init__(
            data_paths=data_paths,
            vocab_path=vocab_path,
            graph_wrapper=graph_wrapper,
            max_seq_len=max_seq_len,
            buf_size=buf_size,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            token_mode=token_mode,
            is_predict=is_predict)

        self.poi_neigh = parse_poi_neigh(graph_path)
        self.avoid_leak = avoid_leak

    def batch_fn(self, batch_examples):
        # batch preprocess 
        batch_texts = []
        batch_query_index = []
        batch_node_type = []
        batch_query_city = []
        batch_poi_index = []
        batch_city_id = []
        batch_addrs = []
        cc = 0

        edges = []
        for ex in batch_examples:
            query = ex.query
            tokens = generate_tokens(query, self.tokenizer)
            batch_texts.append(tokens)
            batch_query_index.append(cc)
            batch_query_city.append(self.city_vocab[ex.city])
            batch_node_type.append(0)
            edges.append((cc, cc))
            cc += 1

        for n, ex in enumerate(batch_examples):
            poi = ex.poi
            pid, city_id, text, addr = parse_poi(poi)
            # city id
            batch_city_id.append(self.city_vocab[city_id])

            tokens = generate_tokens(text, self.tokenizer)
            addr_tokens = generate_tokens(addr, self.tokenizer)
            batch_addrs.append(addr_tokens)
            batch_texts.append(tokens)
            batch_node_type.append(1)
            batch_poi_index.append(cc)
            edges.append((cc, cc))
            poi_cc = cc
            cc += 1
            # Fetch POI-Query Neighbors
            if pid in self.poi_neigh:
                for node_type, q in self.poi_neigh[pid]:
                    if self.avoid_leak and (q == batch_examples[n].query):
                        continue
                    tokens = generate_tokens(q, self.tokenizer)
                    batch_texts.append(tokens)
                    batch_node_type.append(node_type)
                    # build query2poi graph
                    edges.append((cc, poi_cc))
                    cc += 1

        batch = self._pad_batch_records(batch_texts)
        node_feat_dict = {}
        node_feat_dict["src_ids"] = batch[0]
        node_feat_dict["sent_ids"] = batch[1]
        node_feat_dict["pos_ids"] = batch[2]
        node_feat_dict["input_mask"] = batch[3]
        node_feat_dict["node_types"] = np.array(batch_node_type, dtype="int32")

        # Graph Modeling
        g = pgl.graph.Graph(
            num_nodes=len(batch_texts), edges=edges, node_feat=node_feat_dict)
        feed_dict = self.graph_wrapper.to_feed(g)
        batch_labels = [ex.label for ex in batch_examples]
        feed_dict["labels"] = np.array(batch_labels, dtype="float32")
        feed_dict["query_index"] = np.array(batch_query_index, dtype="int64")
        feed_dict["query_city"] = np.array(batch_query_city, dtype="int64")
        feed_dict["poi_index"] = np.array(batch_poi_index, dtype="int64")
        feed_dict["city_id"] = np.array(batch_city_id, dtype="int64")

        if self.is_predict:
            batch_query = [ex.query for ex in batch_examples]
            batch_poi = [ex.poi for ex in batch_examples]
            batch_city = [ex.city for ex in batch_examples]
            feed_dict["batch_query"] = np.array(batch_query)
            feed_dict["batch_poi"] = np.array(batch_poi)
            feed_dict["batch_city"] = np.array(batch_city)
            feed_dict["batch_qid"] = np.array([ ex.qid for ex in batch_examples]) 

        batch = self._pad_batch_records(batch_addrs)
        feed_dict["addr_src_ids"] = batch[0]
        feed_dict["addr_sent_ids"] = batch[1]
        feed_dict["addr_pos_ids"] = batch[2]
        feed_dict["addr_input_mask"] = batch[3]

        return feed_dict
