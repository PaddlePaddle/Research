"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
"""

import math
import numpy
import paddle
import paddle.fluid as fluid
from paddle.fluid import layers
import random
import sys
import time
import os
from common import *


class Constgat(object):
    """
    model class
    """

    def __init__(self, method, base_lr=0.05, fc_lr=0.01, reg=0.0001, is_fleet=False):
        self.method = method
        self.is_fleet = is_fleet

        self.base_lr = base_lr
        self.fc_lr = fc_lr
        self.reg = reg

        self.hidden_dim = 32
        self.output_hidden_dim = 64

        self.duration_class_num = 281

        self.history_context_features = [ \
                ('departure_hour', 'hour')]

        self.link_info_features = [ \
                ('link', 'link'), \
                ('length', 'length')]

        self.start_node_info_features = [ \
                ('start_node', 'node')]

        self.end_node_info_features = [ \
                ('end_node', 'node')]

        self.future_context_features = [ \
                ('future_hour', 'hour')]

        self.neighbor_link_info_features = [ \
                ('neighbor_link', 'link'), \
                ('neighbor_length', 'length')]

        self.neighbor_start_node_info_features = [ \
                ('neighbor_start_node', 'node')]

        self.neighbor_end_node_info_features = [ \
                ('neighbor_end_node', 'node')]

        self.neighbor_link_state_features = [ \
                ('neighbor_duration', 'duration')]

        self.feature_voc_num_dict = { \
                'link': [400000, 8], \
                'length': [2000, 8], \
                'node': [200000, 8], \
                'hour': [24, 8], \
                'duration': [self.duration_class_num, 8]}

    def create_tensors(self, feature_num, name, shape, dtype):
        """
        create tensors
        """
        tensors = []
        for i in range(feature_num):
            tensors.append(
                    layers.data(
                        name='%s_%d' % (name, i),
                        shape=shape,
                        dtype=dtype,
                        lod_level=1))
        return tensors

    def prepare_inputs(self):
        """
        prepare inputs
        """
        history_context = self.create_tensors(
                CONTEXT_FEATURE_NUM, 'history_context', [1], 'int64')
        link_info = self.create_tensors(
                LINK_INFO_FEATURE_NUM, 'link_info', [1], 'int64')
        start_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'start_node_info', [1], 'int64')
        end_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'end_node_info', [1], 'int64')
        future_context = self.create_tensors(
                CONTEXT_FEATURE_NUM, 'future_context', [1], 'int64')
        neighbor_link_info = self.create_tensors(
                LINK_INFO_FEATURE_NUM, 'neighbor_link_info', [1], 'int64')
        neighbor_start_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'neighbor_start_node_info', [1], 'int64')
        neighbor_end_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'neighbor_end_node_info', [1], 'int64')
        neighbor_link_state = self.create_tensors(
                LINK_STATE_FEATURE_NUM, 'neighbor_link_state', [1], 'int64')
        label = layers.data(
                name='label', shape=[1], dtype='int64', lod_level=1)

        return {'history_context': history_context, \
                'link_info': link_info, \
                'start_node_info': start_node_info, \
                'end_node_info': end_node_info, \
                'future_context': future_context, \
                'neighbor_link_info': neighbor_link_info, \
                'neighbor_start_node_info': neighbor_start_node_info, \
                'neighbor_end_node_info': neighbor_end_node_info, \
                'neighbor_link_state': neighbor_link_state, \
                'label': label, \
        }

        return inputs

    def prepare_emb(self, feature_group, feature_info, out_size=0):
        """
        prepare embedding
        """
        embs = []
        for (i, feature) in enumerate(feature_info):
            emb = layers.embedding(
                    input=feature_group[i],
                    param_attr=fluid.ParamAttr(name='%s_emb' % feature[1]),
                    size=self.feature_voc_num_dict[feature[1]],
                    is_sparse=True)
            embs.append(emb)
        concat_emb = layers.concat(embs, axis=1)
        concat_emb = layers.softsign(concat_emb)

        if out_size > 0:
            concat_emb = layers.fc(
                    input=concat_emb,
                    size=out_size,
                    param_attr=fluid.ParamAttr(learning_rate=self.fc_lr),
                    act='relu')

        return concat_emb

    def prepare_preds(self, feature):
        """
        prepare predictions
        """
        hidden1 = layers.fc(
                input=feature,
                size=self.output_hidden_dim,
                param_attr=fluid.ParamAttr(learning_rate=self.fc_lr),
                act='relu')

        hidden2 = layers.fc(
                input=hidden1,
                size=self.output_hidden_dim,
                param_attr=fluid.ParamAttr(learning_rate=self.fc_lr),
                act='relu')

        pred = layers.fc(
                input=hidden2,
                size=1,
                param_attr=fluid.ParamAttr(learning_rate=self.fc_lr),
                act=None)

        return pred

    def prepare_preds_with_name(self, feature, name=''):
        """
        prepare predictions
        """
        hidden1 = layers.fc(
                input=feature,
                size=self.output_hidden_dim,
                param_attr=fluid.ParamAttr(name + '_fc1', learning_rate=self.fc_lr),
                act='relu')

        hidden2 = layers.fc(
                input=hidden1,
                size=self.output_hidden_dim,
                param_attr=fluid.ParamAttr(name + '_fc2', learning_rate=self.fc_lr),
                act='relu')

        pred = layers.fc(
                input=hidden2,
                size=1,
                param_attr=fluid.ParamAttr(name + '_fc3', learning_rate=self.fc_lr),
                act=None)

        return pred

    def prepare_features(self, inputs):
        """
        prepare features
        """
        history_context_concat_emb = self.prepare_emb(
                inputs['history_context'], self.history_context_features, 8)
        link_info_concat_emb = self.prepare_emb(
                inputs["link_info"], self.link_info_features, 16)
        start_node_info_concat_emb = self.prepare_emb(
                inputs["start_node_info"], self.start_node_info_features, 8)
        end_node_info_concat_emb = self.prepare_emb(
                inputs["end_node_info"], self.end_node_info_features, 8)
        future_context_concat_emb = self.prepare_emb(
                inputs["future_context"], self.future_context_features, 8)
        neighbor_link_info_concat_emb = self.prepare_emb(
                inputs["neighbor_link_info"], self.neighbor_link_info_features, 16)
        neighbor_start_node_info_concat_emb = self.prepare_emb(
                inputs["neighbor_start_node_info"], self.neighbor_start_node_info_features, 8)
        neighbor_end_node_info_concat_emb = self.prepare_emb(
                inputs["neighbor_end_node_info"], self.neighbor_end_node_info_features, 8)
        neighbor_link_state_concat_emb = self.prepare_emb(
                inputs["neighbor_link_state"], self.neighbor_link_state_features, 8)

        return {'history_context_concat_emb': history_context_concat_emb, \
                'link_info_concat_emb': link_info_concat_emb, \
                'start_node_info_concat_emb': start_node_info_concat_emb, \
                'end_node_info_concat_emb': end_node_info_concat_emb, \
                'future_context_concat_emb': future_context_concat_emb, \
                'neighbor_link_info_concat_emb': neighbor_link_info_concat_emb, \
                'neighbor_start_node_info_concat_emb': neighbor_start_node_info_concat_emb, \
                'neighbor_end_node_info_concat_emb': neighbor_end_node_info_concat_emb, \
                'neighbor_link_state_concat_emb': neighbor_link_state_concat_emb}

    def attention(self, query_feature, key_feature, value_feature, hidden_dim, name):
        """
        attention
        """
        query_fc = layers.fc(
                input=query_feature,
                size=hidden_dim,
                param_attr=fluid.ParamAttr(name='query_fc_%s' % name, learning_rate=self.fc_lr),
                act='relu',
                num_flatten_dims=2)

        key_fc = layers.fc(
                input=key_feature,
                size=hidden_dim,
                param_attr=fluid.ParamAttr('key_fc_%s' % name, learning_rate=self.fc_lr),
                act='relu',
                num_flatten_dims=2)

        value_fc = layers.fc(
                input=value_feature,
                size=hidden_dim,
                param_attr=fluid.ParamAttr('value_fc_%s' % name, learning_rate=self.fc_lr),
                act='relu',
                num_flatten_dims=2)

        query_key_mat = layers.matmul(query_fc, key_fc, False, True)
        query_key_mat = layers.scale(query_key_mat,
                scale=1.0 / math.sqrt(hidden_dim))
        matching_score = layers.softmax(query_key_mat, axis=2)
        attention = layers.matmul(matching_score, value_fc)
        attention

    def constgat(self, inputs):
        """
        constgat
        """
        history_context = inputs["history_context"]
        link_info = inputs["link_info"]
        start_node_info = inputs["start_node_info"]
        end_node_info = inputs["end_node_info"]
        future_context = inputs["future_context"]
        neighbor_link_info = inputs["neighbor_link_info"]
        neighbor_start_node_info = inputs["neighbor_start_node_info"]
        neighbor_end_node_info = inputs["neighbor_end_node_info"]
        neighbor_link_state = inputs["neighbor_link_state"]

        query_feature = layers.concat([
            link_info,
            start_node_info,
            end_node_info,
            future_context],
            axis=1)
        query_feature_reshape = layers.reshape(query_feature, shape=[-1, 1, query_feature.shape[-1]])

        neighbor_features = layers.concat([
            neighbor_link_info,
            neighbor_start_node_info,
            neighbor_end_node_info],
            axis=1)
        neighbor_features = layers.reshape(
                neighbor_features,
                shape=[-1, MAX_NEIGHBOR_NUM, neighbor_features.shape[-1]])
        neighbor_features = layers.expand(
                x=neighbor_features, expand_times=[1, 1, SEQUENCE_LENGTH])
        neighbor_features = layers.reshape(
                neighbor_features,
                shape=[-1, neighbor_features.shape[1] * SEQUENCE_LENGTH, \
                        int(neighbor_features.shape[2] / SEQUENCE_LENGTH)])

        history_context = layers.reshape(
                history_context,
                shape=[-1, SEQUENCE_LENGTH, history_context.shape[-1]])
        history_context = layers.expand(
                x=history_context, expand_times=[1, MAX_NEIGHBOR_NUM, 1])
        '''
        history_context = layers.sequence_expand(
                history_context,
                link_info)
        '''

        neighbor_link_state = layers.reshape(
                neighbor_link_state,
                shape=[-1, MAX_NEIGHBOR_NUM * SEQUENCE_LENGTH,
                    neighbor_link_state.shape[-1]])

        key_feature = layers.concat([
            neighbor_features,
            history_context,
            neighbor_link_state],
            axis=2)

        value_feature = key_feature

        attention = self.attention(query_feature_reshape, key_feature, value_feature, self.hidden_dim, 'gat')
        attention = layers.reshape(attention, shape=[-1, attention.shape[-1]])

        out_features = layers.concat([query_feature, attention], axis=1)

        return out_features

    def constgat_model(self):
        """constgat model"""
        inputs = self.prepare_inputs()
        
        features = self.prepare_features(inputs)
        history_context_concat_emb = features['history_context_concat_emb']
        link_info_concat_emb = features['link_info_concat_emb']
        start_node_info_concat_emb = features['start_node_info_concat_emb']
        end_node_info_concat_emb = features['end_node_info_concat_emb']
        future_context_concat_emb = features['future_context_concat_emb']
        neighbor_link_info_concat_emb = features['neighbor_link_info_concat_emb']
        neighbor_start_node_info_concat_emb = features['neighbor_start_node_info_concat_emb']
        neighbor_end_node_info_concat_emb = features['neighbor_end_node_info_concat_emb']
        neighbor_link_state_concat_emb = features['neighbor_link_state_concat_emb']
        label = features['label']
        
        x = self.constgat({
            'history_context': history_context_concat_emb,
            'link_info': link_info_concat_emb,
            'start_node_info': start_node_info_concat_emb,
            'end_node_info': end_node_info_concat_emb,
            'future_context': future_context_concat_emb,
            'neighbor_link_info': neighbor_link_info_concat_emb,
            'neighbor_start_node_info': neighbor_start_node_info_concat_emb,
            'neighbor_end_node_info': neighbor_end_node_info_concat_emb,
            'neighbor_link_state': neighbor_link_state_concat_emb})

        pred = self.prepare_preds_with_name(x, 'out_pred')
        label = layers.scale(label, scale=0.01)
        loss = layers.huber_loss(pred, label, 1.0)
        loss = layers.mean(loss)
        return pred, label, loss

    def train(self):
        """
        train
        """
        pred, label, loss = self.constgat_model()

        loss.persistable = True

        optimizer = fluid.optimizer.SGD(
                learning_rate=self.base_lr,
                regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=self.reg))
        if self.is_fleet:
            import paddle.fluid.incubate.fleet.geo_parameter_server as fleet
            fleet.init()
            optimizer = fleet.DistributedOptimizer(optimizer)
        optimizer.minimize(loss)

        return pred, label, loss
