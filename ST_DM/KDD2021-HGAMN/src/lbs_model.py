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
"""lbs_model"""
import os
import re
import time
import logging
from random import random
from functools import reduce, partial

import numpy as np
import multiprocessing

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
from pgl.graph_wrapper import GraphWrapper
from pgl.layers.conv import gcn, gat

from model.ernie import ErnieModel, ErnieConfig
from model.erniesage import ErnieSageModel, ErnieSageConfig
from model.cnn import CnnModel, CnnConfig
from utils.args import print_arguments, check_cuda, prepare_logger
from utils.init import init_checkpoint, init_pretraining_params
from pgl.utils import paddle_helper


def ernie_unsqueeze(tensor):
    """ernie_unsqueeze"""
    tensor = L.unsqueeze(tensor, axes=2)
    tensor.stop_gradient = True
    return tensor

class FakeGraphWrapper(object):
    """FakeGraphWrapper"""
    def __init__(self, node_feat):
        self.node_feat = {}
        self.holder_list = []
        for key, shape, dtype in node_feat:
            self.node_feat[key] = L.data(name=key, shape=shape, dtype=dtype)
            self.holder_list.append(self.node_feat[key])
 

class BaseGraphErnie(object):
    """Base Graph Model"""

    def __init__(self, args, task):

        candi_tasks =  [ "predict_query", "predict_poi",
            "pointwise", "pairwise", "listwise", "listwise_hinge"]

        if task not in candi_tasks:
            raise ValueError("task %s not in %s" % (task, candi_tasks))

        self.norm_score = args.norm_score
        self.ernie_config = ErnieConfig(args.ernie_config_path)
        self.ernie_config.print_config()

        self.city_size = 20000
        self.hidden_size = 64

        self._holder_list = []


        node_feature = [
            ('src_ids', [None, args.max_seq_len], "int64"),
            ('pos_ids', [None, args.max_seq_len], "int64"),
            ('sent_ids', [None, args.max_seq_len], "int64"),
            ('input_mask', [None, args.max_seq_len], "float32"),
            ('node_types', [None], "int32"),
        ]

        if task != 'predict_query':
            self.graph_wrapper = GraphWrapper(
                name="graph", place=F.CPUPlace(), node_feat=node_feature)
            self._holder_list.extend(self.graph_wrapper.holder_list)
        elif task == "predict_query":
            # This is for save_inference_mode for query
            self.graph_wrapper = FakeGraphWrapper(
                node_feat=node_feature)
            self._holder_list.extend(self.graph_wrapper.holder_list)

        self.build_model(args, task)

    @property
    def holder_list(self):
        """ holder list """ 
        return self._holder_list

    def city_embedding(self, input):
        """ add city_embeddding """
        input = L.unsqueeze(input, axes=-1)
        return L.embedding(
            input,
            #size=(self.city_size, 20),
            size=(self.city_size, self.hidden_size),
            param_attr=F.ParamAttr(name="city_embed"))

    def build_model(self, args, task):
        """ build graph model"""
        self.query_geo = L.data(name="query_geo", shape=[-1, 80], dtype="float32")
        self.holder_list.append(self.query_geo)
        self.poi_geo = L.data(name="poi_geo", shape=[-1, 40], dtype="float32")
        self.holder_list.append(self.poi_geo)

        if task != "predict_query":
            self.city_id = L.data(name="city_id", shape=[-1], dtype="int64")
            self.holder_list.append(self.city_id)

            poi_city_embed = self.city_embedding(self.city_id)

            self.poi_index = L.data(name="poi_index", shape=[-1], dtype="int64")
            self.holder_list.append(self.poi_index)

        if task != "predict_poi":
            self.query_city = L.data(name="query_city", shape=[-1], dtype="int64")
            self.holder_list.append(self.query_city)
            query_city_embed = self.city_embedding(self.query_city)

            self.query_index = L.data(
                name="query_index", shape=[-1], dtype="int64")
            self.holder_list.append(self.query_index)


        if task == 'pointwise':
            self.labels = L.data(name="labels", shape=[-1], dtype="float32")
            self.holder_list.append(self.labels)
        elif task == "pairwise":
            self.labels = L.data(name="labels", shape=[-1], dtype="float32")
            self.holder_list.append(self.labels)
            self.labels = L.reshape(self.labels, [-1, 1])
            self.labels.stop_gradients = True
        elif task == "listwise" or task == "listwise_hinge":
            self.labels = L.data(name="labels", shape=[-1], dtype="int64")
            self.holder_list.append(self.labels)
            self.labels = L.reshape(self.labels, [-1, 1])
            self.labels.stop_gradients = True
        elif task == "predict_query":
            pass
        elif task == "predict_poi":
            pass

        src_ids = self.graph_wrapper.node_feat["src_ids"]
        pos_ids = self.graph_wrapper.node_feat["pos_ids"]
        sent_ids = self.graph_wrapper.node_feat["sent_ids"]
        input_mask = self.graph_wrapper.node_feat["input_mask"]

        src_ids = ernie_unsqueeze(src_ids)
        pos_ids = ernie_unsqueeze(pos_ids)
        sent_ids = ernie_unsqueeze(sent_ids)
        input_mask = ernie_unsqueeze(input_mask)
        task_ids = L.zeros_like(sent_ids)
        task_ids = L.cast(task_ids, dtype="int64")

        if args.model_type == "cnn":
            encoder_model = CnnModel
        elif args.model_type == "ernie":
            encoder_model = ErnieModel 
        else:
            raise ValueError("model type %s not exists." % args.model_type)

        ernie = encoder_model(
                src_ids=src_ids,
                position_ids=pos_ids,
                sentence_ids=sent_ids,
                input_mask=input_mask,
                config=self.ernie_config,
                task_ids=task_ids, )

        if task != "predict_query":
            args.max_addr_len = args.max_seq_len

            addr_src_ids = L.data(
                name='addr_src_ids',
                shape=[None, args.max_addr_len],
                dtype="int64")
            self.holder_list.append(addr_src_ids)

            addr_pos_ids = L.data(
                name='addr_pos_ids',
                shape=[None, args.max_addr_len],
                dtype="int64")
            self.holder_list.append(addr_pos_ids)

            addr_sent_ids = L.data(
                name='addr_sent_ids',
                shape=[None, args.max_addr_len],
                dtype="int64")
            self.holder_list.append(addr_sent_ids)

            addr_input_mask = L.data(
                name='addr_input_mask',
                shape=[None, args.max_addr_len],
                dtype="float32")
            self.holder_list.append(addr_input_mask)

            addr_src_ids = ernie_unsqueeze(addr_src_ids)
            addr_pos_ids = ernie_unsqueeze(addr_pos_ids)
            addr_sent_ids = ernie_unsqueeze(addr_sent_ids)
            addr_input_mask = ernie_unsqueeze(addr_input_mask)
            addr_task_ids = L.zeros_like(addr_sent_ids)
            addr_task_ids = L.cast(addr_task_ids, dtype="int64")

            addr_ernie = encoder_model(
                src_ids=addr_src_ids,
                position_ids=addr_pos_ids,
                sentence_ids=addr_sent_ids,
                input_mask=addr_input_mask,
                config=self.ernie_config,
                task_ids=addr_task_ids, )

            addr_repr = addr_ernie.get_pooled_output()

        # get first token as sentence repr
        sent_repr = ernie.get_pooled_output()

        if task != "predict_poi":
            self.query_repr = L.gather(
                sent_repr, self.query_index, overwrite=False)

            self.query_city_embed = query_city_embed
            for_concat = []
            if args.with_city:
                for_concat.append(query_city_embed)
            if args.with_geo_id:
                for_concat.append(self.query_geo)
            
            if len(for_concat) > 0:
                self.query_repr = L.concat(
                    [self.query_repr ] + for_concat, axis=-1)

            self.query_repr = L.fc(self.query_repr,
                               self.hidden_size,
                               act="tanh",
                               name="query_fc")
            self.query_city_score = L.reduce_sum(L.l2_normalize(self.query_city_embed, -1) *
                                              L.l2_normalize(self.query_repr, -1), -1)

        if task != "predict_query":
            neigh_repr = self.neighbor_aggregator(sent_repr)

            self.poi_repr = L.gather(sent_repr, self.poi_index, overwrite=False)
            for_concat = [self.poi_repr, addr_repr, ]
            if args.with_city:
                for_concat.append(poi_city_embed)

            if args.with_geo_id:
                for_concat.append(self.poi_geo)

            if neigh_repr is not None:
                poi_neigh_repr = L.gather(
                    neigh_repr, self.poi_index, overwrite=False)
                for_concat.append(poi_neigh_repr)

            self.poi_repr = L.concat(for_concat, axis=-1)

            self.poi_repr = L.fc(self.poi_repr,
                             self.hidden_size,
                             act="tanh",
                             name="pos_fc")

        if task == "pointwise":
            self.pointwise_loss()
        elif task == "pairwise":
            self.pairwise_loss()
        elif task == "listwise":
            self.listwise_loss(args)
        elif task == "listwise_hinge":
            self.listwise_hinge_loss()

    def pointwise_loss(self):
        """point wise model"""
        self.logits = L.reduce_sum(self.query_repr * self.poi_repr, -1)
        self.score = L.sigmoid(self.logits)
        self.loss = L.sigmoid_cross_entropy_with_logits(
            L.reshape(self.logits, [-1, 1]), L.reshape(self.labels, [-1, 1]))

        auc_label = L.cast(self.labels, dtype="int64")
        auc_label.stop_gradients = True
        _, self.batch_auc, _ = L.auc(
            L.reshape(self.score, [-1, 1]), L.reshape(auc_label, [-1, 1]))
        self.metrics = [L.reduce_mean(self.loss), self.batch_auc]
        self.loss = L.reduce_mean(self.loss)

    def pairwise_loss(self):
        """pairwise model"""
        poi_repr = L.split(self.poi_repr, 2, dim=0)
        pos_repr, neg_repr = poi_repr
        pos_pred = L.cos_sim(self.query_repr, pos_repr)
        neg_pred = L.cos_sim(self.query_repr, neg_repr)

        mode = 'hinge_loss'
        # log(1 + e-z), max(0, 1 - z)
        if 'hinge_loss' == mode:
            theta_z = L.relu(1 + neg_pred - pos_pred)
        elif 'logistic_loss' == mode:
            theta_z = L.log(1 + L.exp(neg_pred - pos_pred))
        self.loss = L.reduce_mean(theta_z)
        pos_cnt = L.reduce_sum(L.cast(L.greater_than(pos_pred, neg_pred), dtype="float32"))
        neg_cnt = L.reduce_sum(L.cast(L.less_than(pos_pred, neg_pred), dtype="float32"))
        self.order = pos_cnt / (1e-5 + neg_cnt)
        self.metrics = [self.loss, self.order]

    def listwise_loss(self, args):
        """listwise model"""
        self.logits = L.matmul(
            self.query_repr, self.poi_repr, transpose_y=True)
        if self.norm_score:
            self.logits = L.softsign(self.logits)

        if args.scale_softmax:
            scale = L.create_parameter(shape=[1], dtype="float32", name="final_scale", default_initializer=F.initializer.ConstantInitializer(value=1.0))
            bias = L.create_parameter(shape=[1], dtype="float32", name="final_bias", default_initializer=F.initializer.ConstantInitializer(value=0.0))
            self.logits = self.logits * scale * scale + bias

        self.score = L.softmax(self.logits)
        self.loss = L.softmax_with_cross_entropy(self.logits, self.labels)
        self.loss = L.reduce_mean(self.loss)
        self.acc = L.accuracy(L.softmax(self.logits), self.labels)
        self.metrics = [self.loss, self.acc]

    def listwise_hinge_loss(self):
        """listwise hinge loss model"""
        self.poi_repr = L.l2_normalize(self.poi_repr, -1)
        self.query_repr = L.l2_normalize(self.query_repr, -1)
        pos_logits = L.reduce_sum(self.query_repr * self.poi_repr, -1, keep_dim=True)
        neg_logits = L.matmul(self.query_repr, self.poi_repr, transpose_y = True)
        self.loss = L.reduce_mean(L.relu(neg_logits - pos_logits + 0.3))
        self.acc = L.accuracy(L.softmax(neg_logits), self.labels)
        self.metrics = [self.loss, self.acc]

    def neighbor_aggregator(self, sent_repr):
        """neighbor aggregation"""
        return None


class GCNGraphErnie(BaseGraphErnie):
    """ GCN Graph Ernie"""

    def neighbor_aggregator(self, sent_repr):
        #norm = L.clamp(L.reshape(L.cast(self.graph_wrapper.indegree(), dtype="float32"), [-1, 1]), min=1.)
        norm = L.ones_like(sent_repr)
        def send_func(src, dst , edge):
            return src["h"]
        msg = self.graph_wrapper.send(send_func, nfeat_list=[("h", norm)])
        norm = self.graph_wrapper.recv(msg, "sum")
        norm = L.reduce_mean(norm, -1, keep_dim=True)
        norm = L.clamp(norm, min=1.0)
        
        return gcn(self.graph_wrapper,
                   sent_repr,
                   self.hidden_size,
                   activation="relu",
                   name="gcn") / norm

class GATNEGraphErnie(BaseGraphErnie):
    """ GATNE Graph Ernie"""

    def neighbor_aggregator(self, sent_repr):
        def send_func(src, dst , edge):
            return { "h": src["h"], "nt": src["node_type"], "att": src["att"] }

        def recv_func(message):
            nt = message["nt"]
            att = message["att"]
            h = message["h"]
            output_h = []
            for i in range(2):
                mask = L.cast(nt == i, dtype="float32") 
                rel_att = att[:, i:i+1] + ( 1 - mask ) * -10000
                rel_att = paddle_helper.sequence_softmax(rel_att)
                rel_h = L.sequence_pool(h * rel_att * mask, "sum")
                output_h.append(rel_h)
            output_h = L.concat(output_h, -1)
            return output_h

        attention = L.fc(sent_repr, size=2, name="attention")
        msg = self.graph_wrapper.send(send_func, nfeat_list=[ ("h", sent_repr), ("node_type", self.graph_wrapper.node_feat["node_types"] ), ("att", attention)])
        output = self.graph_wrapper.recv(msg, recv_func)
        return output


class GATGraphErnie(BaseGraphErnie):
    """ GCN Graph Ernie"""

    def neighbor_aggregator(self, sent_repr):
        return gat(self.graph_wrapper,
                   sent_repr,
                   self.hidden_size // 8, 
                   activation="relu",
                   num_heads=8,
                   name="gat")


class HGCMNGraphErnie(BaseGraphErnie):
    """ HGCMNGraph Graph Ernie"""

    def neighbor_aggregator(self, sent_repr):
        def send_func(src, dst , edge):
            q = dst["query"]
            k = src["key"]
            attn = L.reduce_sum(q * k, -1)
            return { "h": src["h"], "nt": src["node_type"], "att": attn }

        def recv_func(message):
            nt = message["nt"]
            att = message["att"]
            h = message["h"]
            output_h = []
            for i in range(2):
                mask = L.cast(nt == i, dtype="float32") 
                rel_att = att[:, i:i+1] + ( 1 - mask ) * -10000
                rel_att = paddle_helper.sequence_softmax(rel_att)
                rel_h = L.sequence_pool(h * rel_att * mask, "sum")
                output_h.append(rel_h)
            output_h = L.concat(output_h, -1)
            return output_h

        query = L.fc(sent_repr, size=self.hidden_size * 2, name="attn_query") / np.sqrt(self.hidden_size)
        query = L.reshape(query, [-1, 2, self.hidden_size])
        key = L.fc(sent_repr, size=self.hidden_size * 2, name="attn_key")
        key = L.reshape(key, [-1, 2, self.hidden_size])
        msg = self.graph_wrapper.send(send_func,
                 nfeat_list=[ ("h", sent_repr), ("query", query), ("key", key), ("node_type", self.graph_wrapper.node_feat["node_types"] ) ])
        output = self.graph_wrapper.recv(msg, recv_func)
        return output


