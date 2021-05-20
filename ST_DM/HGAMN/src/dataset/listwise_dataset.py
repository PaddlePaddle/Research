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
"""listwise generator"""

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
from dataset.pointwise_dataset import generate_tokens, parse_poi_neigh, parse_poi, CityVocab
import collections
import logging
import tokenization
from batching import pad_batch_data_maxlen
log = logging.getLogger(__name__)
import geohash

import pgl


def semantic_similar(a, b):
    a = [ w.strip() for w in a.split('\x03')[-1].split('\x04') ]
    b = [ w.strip() for w in b.split('\x03')[-1].split('\x04') ]
    a = "".join(a)
    b = "".join(b)
    
    if a == b:
        return True
    else:
        return False
    

def listwise_parse_from_files(data_paths):
    """parse data from files"""

    Example = namedtuple('Example', ["city", 'query', "query_geo",  'pos', 'neg'])

    example_list = []
    for path in data_paths:
        with open(path, mode='r', encoding="utf-8", errors='ignore') as f:
            for line in tqdm.tqdm(f):
                line = line.strip('\r\n').split('\t')
                if len(line) != 5:
                    continue
                qid = line[0]
                qid_info = qid.split('_')
                if len(qid_info) != 6:
                    continue
                cuid, time, loc_cityid, bound_cityid, loc, bound = qid.split('_')
                city = loc_cityid
                query_geo = geohash.get_loc_bound_gh(loc, bound) 
                query = line[1]
                pos_poi = line[2]
                neg_poi = line[3]
                pos_poi = set(pos_poi.split("\x01"))
                neg_poi = set(neg_poi.split("\x01"))
                neg_poi = neg_poi - pos_poi

                neg_poi = [
                    poi for poi in neg_poi if len(poi.split("\x02")) == 6
                ]
                
                # no in-batch neg
                if len(neg_poi) == 0:
                    continue

                neg_poi = np.random.choice(neg_poi, size=5)
                # no in-batch neg

                for poi in pos_poi:
                    if len(poi.split("\x02")) != 6:
                        continue
                    example_list.append(
                        Example(
                            city=int(city), query=query, query_geo=query_geo, pos=poi, neg=neg_poi))
    return example_list


class DataGenerator(BaseDataGenerator):
    """DataGenerator"""

    def __init__(self,
                 data_paths,
                 vocab_path,
                 graph_wrapper,
                 max_seq_len=20,
                 buf_size=1000,
                 num_workers=1,
                 batch_size=128,
                 city_vocab="cities.vocab",
                 token_mode="ernie",
                 is_predict=False):

        super(DataGenerator, self).__init__(
            buf_size=buf_size, num_workers=num_workers, batch_size=batch_size)

        self.graph_wrapper = graph_wrapper
        self.line_examples = listwise_parse_from_files(data_paths)
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
        """batch_fn"""
        batch_texts = []
        batch_query_city = []
        batch_query_index = []
        batch_poi_index = []
        batch_city_id = []
        batch_labels = []
        batch_addrs = []
        cc = 0

        edges = []

        for ex in batch_examples:
            tokens = generate_tokens(ex.query, self.tokenizer)
            batch_texts.append(tokens)
            batch_query_index.append(cc)
            batch_query_city.append(self.city_vocab[ex.city])
            edges.append((cc, cc))
            cc += 1

        for lb, ex in enumerate(batch_examples):
            poi = ex.pos
            pid, city_id, text, addr = parse_poi(poi)
            # city id
            batch_city_id.append(self.city_vocab[city_id])
            batch_labels.append(lb)
            tokens = generate_tokens(text, self.tokenizer)
            batch_texts.append(tokens)
            batch_poi_index.append(cc)
            edges.append((cc, cc))

            addr_tokens = generate_tokens(addr, self.tokenizer)
            batch_addrs.append(addr_tokens)
            poi_cc = cc
            cc += 1

        for lb, ex in enumerate(batch_examples):
            for poi in ex.neg:
                pid, city_id, text, addr = parse_poi(poi)
                # city id
                batch_city_id.append(self.city_vocab[city_id])
                tokens = generate_tokens(text, self.tokenizer)
                batch_texts.append(tokens)
                batch_poi_index.append(cc)
                edges.append((cc, cc))

                addr_tokens = generate_tokens(addr, self.tokenizer)
                batch_addrs.append(addr_tokens)

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
        feed_dict["labels"] = np.array(batch_labels, dtype="int64")
        feed_dict["query_index"] = np.array(batch_query_index, dtype="int64")
        feed_dict["query_city"] = np.array(batch_query_city, dtype="int64")
        feed_dict["poi_index"] = np.array(batch_poi_index, dtype="int64")
        feed_dict["city_id"] = np.array(batch_city_id, dtype="int64")

        batch = self._pad_batch_records(batch_addrs)
        feed_dict["addr_src_ids"] = batch[0]
        feed_dict["addr_sent_ids"] = batch[1]
        feed_dict["addr_pos_ids"] = batch[2]
        feed_dict["addr_input_mask"] = batch[3]

        return feed_dict


class GraphDataGenerator(DataGenerator):
    """ Graph Data Generator"""

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
                 city_vocab="cities.vocab",
                 token_mode="ernie",
                 is_predict=False):

        super(GraphDataGenerator, self).__init__(
            data_paths=data_paths,
            vocab_path=vocab_path,
            graph_wrapper=graph_wrapper,
            max_seq_len=max_seq_len,
            buf_size=buf_size,
            num_workers=num_workers,
            city_vocab="cities.vocab",
            token_mode=token_mode,
            batch_size=batch_size,
            is_predict=is_predict)

        self.poi_neigh = parse_poi_neigh(graph_path)
        self.avoid_leak = avoid_leak

    def batch_fn(self, batch_examples):
        """batch fn"""
        # node: types 0 for query 1 for poi
        batch_texts = []
        batch_query_index = []
        batch_node_type = []
        batch_query_city = []
        batch_poi_index = []
        batch_city_id = []
        batch_labels = []
        batch_addrs = []
        batch_query_geo = []
        batch_poi_geo = []
        cc = 0

        edges = []

        for ex in batch_examples:
            tokens = generate_tokens(ex.query, self.tokenizer)
            batch_texts.append(tokens)
            batch_query_index.append(cc)
            batch_query_city.append(self.city_vocab[ex.city])
            batch_query_geo.append(ex.query_geo)
            edges.append((cc, cc))
            batch_node_type.append(0)
            cc += 1

        for n, ex in enumerate(batch_examples):
            poi = ex.pos

            pid, city_id, text, addr, poi_geo = parse_poi(poi)
            # city id
            batch_city_id.append(self.city_vocab[city_id])

            batch_labels.append(n)

            addr_tokens = generate_tokens(addr, self.tokenizer)
            batch_addrs.append(addr_tokens)
            batch_poi_geo.append(poi_geo)

            tokens = generate_tokens(text, self.tokenizer)
            batch_texts.append(tokens)
            batch_poi_index.append(cc)
            batch_node_type.append(1)
            edges.append((cc, cc))
            poi_cc = cc
            cc += 1
            # Fetch POI-Query Neighbors
            if pid in self.poi_neigh:
                for node_type, q in self.poi_neigh[pid]:
                    if self.avoid_leak and semantic_similar(q, batch_examples[n].query):
                        continue
                    tokens = generate_tokens(q, self.tokenizer)
                    batch_texts.append(tokens)
                    batch_node_type.append(node_type)
                    # build query2poi graph
                    edges.append((cc, poi_cc))
                    cc += 1

        for n, ex in enumerate(batch_examples):
            for poi in ex.neg:
                pid, city_id, text, addr, poi_geo = parse_poi(poi)
                # city id
                batch_city_id.append(self.city_vocab[city_id])

                tokens = generate_tokens(text, self.tokenizer)
                batch_texts.append(tokens)
                batch_poi_index.append(cc)
                batch_poi_geo.append(poi_geo)
                batch_node_type.append(1)

                addr_tokens = generate_tokens(addr, self.tokenizer)
                batch_addrs.append(addr_tokens)

                edges.append((cc, cc))
                poi_cc = cc
                cc += 1
                # Fetch POI-Query Neighbors
                if pid in self.poi_neigh:
                    for node_type, q in self.poi_neigh[pid]:
                        if self.avoid_leak and semantic_similar(q, batch_examples[n].query):
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
        feed_dict["labels"] = np.array(batch_labels, dtype="int64")
        feed_dict["query_index"] = np.array(batch_query_index, dtype="int64")
        feed_dict["query_city"] = np.array(batch_query_city, dtype="int64")
        feed_dict["query_geo"] = np.array(batch_query_geo, dtype="float32")
        feed_dict["poi_geo"] = np.array(batch_poi_geo, dtype="float32")
        feed_dict["poi_index"] = np.array(batch_poi_index, dtype="int64")
        feed_dict["city_id"] = np.array(batch_city_id, dtype="int64")

        batch = self._pad_batch_records(batch_addrs)
        feed_dict["addr_src_ids"] = batch[0]
        feed_dict["addr_sent_ids"] = batch[1]
        feed_dict["addr_pos_ids"] = batch[2]
        feed_dict["addr_input_mask"] = batch[3]

        return feed_dict
