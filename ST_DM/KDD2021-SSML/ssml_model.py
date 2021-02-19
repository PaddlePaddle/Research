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


class SSML(object):
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

        self.support_history_context_features = [ \
                ('support_departure_hour', 'hour')]

        self.support_link_info_features = [ \
                ('support_link', 'link'), \
                ('support_length', 'length')]

        self.support_start_node_info_features = [ \
                ('support_start_node', 'node')]

        self.support_end_node_info_features = [ \
                ('support_end_node', 'node')]

        self.support_future_context_features = [ \
                ('support_future_hour', 'hour')]

        self.support_neighbor_link_info_features = [ \
                ('support_neighbor_link', 'link'), \
                ('support_neighbor_length', 'length')]

        self.support_neighbor_start_node_info_features = [ \
                ('support_neighbor_start_node', 'node')]

        self.support_neighbor_end_node_info_features = [ \
                ('support_neighbor_end_node', 'node')]

        self.support_neighbor_link_state_features = [ \
                ('support_neighbor_duration', 'duration')]

        self.support_duration_delta = [ \
                ('support_duration_delta_discrete', 'delta')]

        self.query_history_context_features = [ \
                ('query_departure_hour', 'hour'), \
                ('query_history_context', 'time')]

        self.query_link_info_features = [ \
                ('query_link', 'link'), \
                ('query_length', 'length')]

        self.query_start_node_info_features = [ \
                ('query_start_node', 'node')]

        self.query_end_node_info_features = [ \
                ('query_end_node', 'node')]

        self.query_future_context_features = [ \
                ('query_future_hour', 'hour')]

        self.query_neighbor_link_info_features = [ \
                ('query_neighbor_link', 'link'), \
                ('query_neighbor_length', 'length')]

        self.query_neighbor_start_node_info_features = [ \
                ('query_neighbor_start_node', 'node')]

        self.query_neighbor_end_node_info_features = [ \
                ('query_neighbor_end_node', 'node')]

        self.query_neighbor_link_state_features = [ \
                ('query_neighbor_duration', 'duration')]

        self.feature_voc_num_dict = { \
                'link': [400000, 8], \
                'length': [2000, 8], \
                'node': [200000, 8], \
                'hour': [24, 8], \
                'duration': [self.duration_class_num, 8],                
                'delta': [300, 8]}

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
        support_history_context = self.create_tensors(
                CONTEXT_FEATURE_NUM, 'support_history_context', [1], 'int64')
        support_link_info = self.create_tensors(
                LINK_INFO_FEATURE_NUM, 'support_link_info', [1], 'int64')
        support_start_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'support_start_node_info', [1], 'int64')
        support_end_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'support_end_node_info', [1], 'int64')
        support_future_context = self.create_tensors(
                CONTEXT_FEATURE_NUM, 'support_future_context', [1], 'int64')
        support_neighbor_link_info = self.create_tensors(
                LINK_INFO_FEATURE_NUM, 'support_neighbor_link_info', [1], 'int64')
        support_neighbor_start_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'support_neighbor_start_node_info', [1], 'int64')
        support_neighbor_end_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'support_neighbor_end_node_info', [1], 'int64')
        support_neighbor_link_state = self.create_tensors(
                LINK_STATE_FEATURE_NUM, 'support_neighbor_link_state', [1], 'int64')
        support_mask = layers.data(
                name='support_mask', shape=[1], dtype='int64', lod_level=1)
        support_duration_delta = layers.data(
                name='support_duration_delta', shape=[1], dtype='int64', lod_level=1)

        query_history_context = self.create_tensors(
                CONTEXT_FEATURE_NUM, 'query_history_context', [1], 'int64')
        query_link_info = self.create_tensors(
                LINK_INFO_FEATURE_NUM, 'query_link_info', [1], 'int64')
        query_start_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'query_start_node_info', [1], 'int64')
        query_end_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'query_end_node_info', [1], 'int64')
        query_future_context = self.create_tensors(
                CONTEXT_FEATURE_NUM, 'query_future_context', [1], 'int64')
        query_neighbor_link_info = self.create_tensors(
                LINK_INFO_FEATURE_NUM, 'query_neighbor_link_info', [1], 'int64')
        query_neighbor_start_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'query_neighbor_start_node_info', [1], 'int64')
        query_neighbor_end_node_info = self.create_tensors(
                NODE_INFO_FEATURE_NUM, 'query_neighbor_end_node_info', [1], 'int64')
        query_neighbor_link_state = self.create_tensors(
                LINK_STATE_FEATURE_NUM, 'query_neighbor_link_state', [1], 'int64')
        query_mask = layers.data(
                name='query_mask', shape=[1], dtype='int64', lod_level=1)
        query_duration_delta = layers.data(
                name='query_duration_delta', shape=[1], dtype='int64', lod_level=1)

        return {'support_history_context': support_history_context, \
                'support_link_info': support_link_info, \
                'support_start_node_info': support_start_node_info, \
                'support_end_node_info': support_end_node_info, \
                'support_future_context': support_future_context, \
                'support_neighbor_link_info': support_neighbor_link_info, \
                'support_neighbor_start_node_info': support_neighbor_start_node_info, \
                'support_neighbor_end_node_info': support_neighbor_end_node_info, \
                'support_neighbor_link_state': support_neighbor_link_state, \
                'support_mask': support_mask, \
                'support_duration_delta': support_duration_delta, \

                'query_history_context': query_history_context, \
                'query_link_info': query_link_info, \
                'query_start_node_info': query_start_node_info, \
                'query_end_node_info': query_end_node_info, \
                'query_future_context': query_future_context, \
                'query_neighbor_link_info': query_neighbor_link_info, \
                'query_neighbor_start_node_info': query_neighbor_start_node_info, \
                'query_neighbor_end_node_info': query_neighbor_end_node_info, \
                'query_neighbor_link_state': query_neighbor_link_state, \
                'query_mask': query_mask, \
                'query_duration_delta': query_duration_delta, \
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
        support_history_context_concat_emb = self.prepare_emb(
                inputs['support_history_context'], self.support_history_context_features, 8)
        support_link_info_concat_emb = self.prepare_emb(
                inputs["support_link_info"], self.support_link_info_features, 16)
        support_start_node_info_concat_emb = self.prepare_emb(
                inputs["support_start_node_info"], self.support_start_node_info_features, 8)
        support_end_node_info_concat_emb = self.prepare_emb(
                inputs["support_end_node_info"], self.support_end_node_info_features, 8)
        support_future_context_concat_emb = self.prepare_emb(
                inputs["support_future_context"], self.support_future_context_features, 8)
        support_neighbor_link_info_concat_emb = self.prepare_emb(
                inputs["support_neighbor_link_info"], self.support_neighbor_link_info_features, 16)
        support_neighbor_start_node_info_concat_emb = self.prepare_emb(
                inputs["support_neighbor_start_node_info"], self.support_neighbor_start_node_info_features, 8)
        support_neighbor_end_node_info_concat_emb = self.prepare_emb(
                inputs["support_neighbor_end_node_info"], self.support_neighbor_end_node_info_features, 8)
        support_neighbor_link_state_concat_emb = self.prepare_emb(
                inputs["support_neighbor_link_state"], self.support_neighbor_link_state_features, 8)
        support_duration_delta_concat_emb = self.prepare_emb(
                [inputs["support_duration_delta"]], self.support_duration_delta, 8)

        query_history_context_concat_emb = self.prepare_emb(
                inputs['query_history_context'], self.query_history_context_features, 8)
        query_link_info_concat_emb = self.prepare_emb(
                inputs["query_link_info"], self.query_link_info_features, 16)
        query_start_node_info_concat_emb = self.prepare_emb(
                inputs["query_start_node_info"], self.query_start_node_info_features, 8)
        query_end_node_info_concat_emb = self.prepare_emb(
                inputs["query_end_node_info"], self.query_end_node_info_features, 8)
        query_future_context_concat_emb = self.prepare_emb(
                inputs["query_future_context"], self.query_future_context_features, 8)
        query_neighbor_link_info_concat_emb = self.prepare_emb(
                inputs["query_neighbor_link_info"], self.query_neighbor_link_info_features, 16)
        query_neighbor_start_node_info_concat_emb = self.prepare_emb(
                inputs["query_neighbor_start_node_info"], self.query_neighbor_start_node_info_features, 8)
        query_neighbor_end_node_info_concat_emb = self.prepare_emb(
                inputs["query_neighbor_end_node_info"], self.query_neighbor_end_node_info_features, 8)
        query_neighbor_link_state_concat_emb = self.prepare_emb(
                inputs["query_neighbor_link_state"], self.query_neighbor_link_state_features, 8)

        return {'support_history_context_concat_emb': support_history_context_concat_emb, \
                'support_link_info_concat_emb': support_link_info_concat_emb, \
                'support_start_node_info_concat_emb': support_start_node_info_concat_emb, \
                'support_end_node_info_concat_emb': support_end_node_info_concat_emb, \
                'support_future_context_concat_emb': support_future_context_concat_emb, \
                'support_neighbor_link_info_concat_emb': support_neighbor_link_info_concat_emb, \
                'support_neighbor_start_node_info_concat_emb': support_neighbor_start_node_info_concat_emb, \
                'support_neighbor_end_node_info_concat_emb': support_neighbor_end_node_info_concat_emb, \
                'support_neighbor_link_state_concat_emb': support_neighbor_link_state_concat_emb, \
                'support_duration_delta_concat_emb': support_duration_delta_concat_emb, \
                'query_history_context_concat_emb': query_history_context_concat_emb, \
                'query_link_info_concat_emb': query_link_info_concat_emb, \
                'query_start_node_info_concat_emb': query_start_node_info_concat_emb, \
                'query_end_node_info_concat_emb': query_end_node_info_concat_emb, \
                'query_future_context_concat_emb': query_future_context_concat_emb, \
                'query_neighbor_link_info_concat_emb': query_neighbor_link_info_concat_emb, \
                'query_neighbor_start_node_info_concat_emb': query_neighbor_start_node_info_concat_emb, \
                'query_neighbor_end_node_info_concat_emb': query_neighbor_end_node_info_concat_emb, \
                'query_neighbor_link_state_concat_emb': query_neighbor_link_state_concat_emb}

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

    def self_attention_model(self):
        """
        self_attention with support/query distinguish
        """
        inputs = self.prepare_inputs()
        
        features = self.prepare_features(inputs)
        support_history_context_concat_emb = features['support_history_context_concat_emb']
        support_link_info_concat_emb = features['support_link_info_concat_emb']
        support_start_node_info_concat_emb = features['support_start_node_info_concat_emb']
        support_end_node_info_concat_emb = features['support_end_node_info_concat_emb']
        support_future_context_concat_emb = features['support_future_context_concat_emb']
        support_neighbor_link_info_concat_emb = features['support_neighbor_link_info_concat_emb']
        support_neighbor_start_node_info_concat_emb = features['support_neighbor_start_node_info_concat_emb']
        support_neighbor_end_node_info_concat_emb = features['support_neighbor_end_node_info_concat_emb']
        support_neighbor_link_state_concat_emb = features['support_neighbor_link_state_concat_emb']
        support_y_embed = features['support_duration_delta_concat_emb']

        query_history_context_concat_emb = features['query_history_context_concat_emb']
        query_link_info_concat_emb = features['query_link_info_concat_emb']
        query_start_node_info_concat_emb = features['query_start_node_info_concat_emb']
        query_end_node_info_concat_emb = features['query_end_node_info_concat_emb']
        query_future_context_concat_emb = features['query_future_context_concat_emb']
        query_neighbor_link_info_concat_emb = features['query_neighbor_link_info_concat_emb']
        query_neighbor_start_node_info_concat_emb = features['query_neighbor_start_node_info_concat_emb']
        query_neighbor_end_node_info_concat_emb = features['query_neighbor_end_node_info_concat_emb']
        query_neighbor_link_state_concat_emb = features['query_neighbor_link_state_concat_emb']
        
        support_x = self.constgat({
            'history_context': support_history_context_concat_emb,
            'link_info': support_link_info_concat_emb,
            'start_node_info': support_start_node_info_concat_emb,
            'end_node_info': support_end_node_info_concat_emb,
            'future_context': support_future_context_concat_emb,
            'neighbor_link_info': support_neighbor_link_info_concat_emb,
            'neighbor_start_node_info': support_neighbor_start_node_info_concat_emb,
            'neighbor_end_node_info': support_neighbor_end_node_info_concat_emb,
            'neighbor_link_state': support_neighbor_link_state_concat_emb})

        query_x = self.constgat({
            'history_context': query_history_context_concat_emb,
            'link_info': query_link_info_concat_emb,
            'start_node_info': query_start_node_info_concat_emb,
            'end_node_info': query_end_node_info_concat_emb,
            'future_context': query_future_context_concat_emb,
            'neighbor_link_info': query_neighbor_link_info_concat_emb,
            'neighbor_start_node_info': query_neighbor_start_node_info_concat_emb,
            'neighbor_end_node_info': query_neighbor_end_node_info_concat_emb,
            'neighbor_link_state': query_neighbor_link_state_concat_emb})
        query_x = query_x + layers.reduce_sum(inputs['query_mask']) * 0.0 + layers.reduce_sum(inputs['support_mask']) * 0.0   # IMPORTANT: for save_inference_model

        def forward_attention(indicator, support_x, support_y_embed, support_mask, query_x, query_y, query_mask):
            """
            support_indicator: length = support_len
                if attention(support, query), indicator = 0
                if attention(support, support), indicator = 1
            """
            support_y_embed = support_y_embed * support_mask
            support_xy = layers.concat([support_x, support_y_embed, indicator], axis=1)

            pad_value = layers.assign(input=numpy.array([0.0], dtype=numpy.float32))
            support_pad, support_len = layers.sequence_pad(support_xy, pad_value=pad_value)
            query_pad, query_len = layers.sequence_pad(query_x, pad_value=pad_value)

            attention = self.attention(query_pad, support_pad, support_pad, self.hidden_dim, 'meta')
            attention = layers.sequence_unpad(attention, length=query_len)
            pred_input = layers.concat([attention, query_x], axis=1)

            pred = self.prepare_preds_with_name(pred_input, 'out_pred')
            label = layers.cast(query_y, dtype='float32')
            label = layers.scale(label, scale=0.01)

            loss = layers.huber_loss(pred, label, 1.0) * query_mask
            loss = layers.mean(loss)
            return pred, label, loss

        indicator = support_y_embed * 0.0
        pred, label, loss1 = forward_attention(
                indicator, support_x, support_y_embed, inputs['support_mask'], 
                query_x, inputs['query_duration_delta'], 1.0)
        indicator = support_y_embed * 1.0
        _, _, loss2 = forward_attention(
                indicator, support_x, support_y_embed, inputs['support_mask'], 
                support_x, inputs['support_duration_delta'] - 120, (inputs['support_mask'] * (-1.0) + 1))
        loss = loss1 + loss2
        return pred, label, loss

    def train(self):
        """
        train
        """
        pred, label, loss = self.self_attention_model()

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
