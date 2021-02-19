#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: nets/common/net_helper.py
Author: map(zhuoan@baidu.com)
Date: 2020/06/15 13:36:32
"""
import math
import numpy as np
import logging
import paddle.fluid as fluid


def ffn(input, d_hid, d_size, name="ffn"):
    """
        Position-wise Feed-Forward Network
        input is LoDTensor
    """
    hidden = fluid.layers.fc(input=input,
             size=d_hid,
             num_flatten_dims=1,
             param_attr=fluid.ParamAttr(name=name + '_innerfc_weight'),
             bias_attr=fluid.ParamAttr(
                 name=name + '_innerfc_bias',
                 initializer=fluid.initializer.Constant(0.)),
             act="leaky_relu")

    out = fluid.layers.fc(input=hidden,
              size=d_size,
              num_flatten_dims=1,
              param_attr=fluid.ParamAttr(name=name + '_outerfc_weight'),
              bias_attr=fluid.ParamAttr(
                  name=name + '_outerfc_bias',
                  initializer=fluid.initializer.Constant(0.)))
    return out


def general_attention(input, dropout_rate=None):
    """
        mode: mlp, concat, general, location
        score(st,hi)=st^t * Wa * hi
    """ 
    #bias_attr = fluid.ParamAttr(
    #    regularizer=fluid.regularizer.L2Decay(0.0),
    #    initializer=fluid.initializer.NormalInitializer(scale=0.0)))

    input_weight = fluid.layers.fc(
        input=input,
        size=1,
        act='sequence_softmax')

    if dropout_rate:
        input_weight = fluid.layers.dropout(
            input_weight, dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)
    scaled = fluid.layers.elementwise_mul(
        x=input, y=input_weight, axis=0)
    att_out = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
    return att_out


def dot_product_attention(query, key, value, d_key, q_mask=None, k_mask=None,
        dropout_rate=None, name=None):
    """
     Args:
         query: a tensor with shape [batch, Q_time, Q_dimension]
         key: a tensor with shape [batch, time, K_dimension]
         value: a tensor with shape [batch, time, V_dimension]

     Returns:
         a tensor with shape [batch, query_time, value_dimension]

     Raises:
         AssertionError: if Q_dimension not equal to K_dimension when attention 
                        type is dot.
    """ 
    logits = fluid.layers.matmul(x=query, y=key, transpose_y=True, alpha=d_key ** (-0.5))

    if (q_mask is not None) and (k_mask is not None):
        mask = fluid.layers.matmul(x=q_mask, y=k_mask, transpose_y=True)
        mask = fluid.layers.scale(
            mask,
            scale=float(2 ** 32 - 1),
            bias=float(-1),
            bias_after_scale=False)
        mask.stop_gradient = True
        logits += mask
    attention = fluid.layers.softmax(logits)
    if dropout_rate:
        attention = fluid.layers.dropout(
            attention, dropout_prob=dropout_rate,
            dropout_implementation="upscale_in_train",
            is_test=False)
    atten_out = fluid.layers.matmul(x=attention, y=value)

    return atten_out


def get_attention_rnn(rnn_encoder_output, rnn_encoder_state, attention_version=0,
            attention_size=None):
    """
    TODO:
    rnn_encoder_output
    [[batch_size, seq_length, num_units] rnn output, [batch_size, seq_length] mask]
    return [batch_size, attention_size]
    """
    encoder_output, seq_mask = rnn_encoder_output
    if attention_version == 0:
        return rnn_encoder_state, None
    elif attention_version == 1:
        #sim to hidden state
        rnn_encoder_state = fluid.layers.expand(rnn_encoder_state, -1)
        score = fluid.squeeze(fluid.matmul(encoder_output, rnn_encoder_state), 2)
        alphas = fluid.softmax(score, 1)
        #context : [batch_size, dim, 1]
        context = fluid.matmul(fluid.transpose(encoder_output, [0, 2, 1]),
                fluid.expand(alphas, 2))
        attention = fluid.squeeze(context, 2) # [batch_size, dim]
        return attention, alphas

    if attention_size is None:
        attention_size = encoder_output.get_shape().as_list()[-1] #get last dimension
    
    #1.one-layer mlp 
    #[batch_size, seq_legnth, attention_size]
    if attention_version == 2:
        g = fluid.layers.dense(rnn_encoder_output, attention_size,
            activation=fluid.tanh, use_bias=True,
            kernel_initializer=fluid.Constant(0),
            name="hidden_trans")
    else:
        g = rnn_encoder_output
     
    #2.compute weight by compute simility of u and attention vector g 
    u = fluid.create_param("hidden_weights", shape=[attention_size],
            initializer=fluid.Constant(0))
    #[batch_size, seq_length, 1]
    score = fluid.reduce_sum(fluid.multiply(g, u), axis=2) #[batch_size, seq_length]
    #mask score
    if seq_mask is not None:
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - seq_mask) * -10000.0
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        score += adder

    alphas = fluid.softmax(score)
    attention = fluid.reduce_sum(rnn_encoder_output * fluid.expand_dims(alphas, -1), axis=1)
    attention = fluid.reshape(attention, [-1, attention_size])
    return attention, alphas


def din_attention(hist, target_expand, mask):
    """
    TODO: activation weight
    user interest
    """
    hidden_size = hist.shape[-1]

    concat = fluid.layers.concat(
        [hist, target_expand, hist - target_expand, hist * target_expand], axis=2)
    atten_fc1 = fluid.layers.fc(name="atten_fc1",
                                input=concat,
                                size=80,
                                act="sigmoid",
                                num_flatten_dims=2)
    atten_fc2 = fluid.layers.fc(name="atten_fc2",
                                input=atten_fc1,
                                size=40,
                                act="sigmoid",
                                num_flatten_dims=2)
    atten_fc3 = fluid.layers.fc(name="atten_fc3",
                                input=atten_fc2,
                                size=1,
                                num_flatten_dims=2)
    atten_fc3 += mask
    atten_fc3 = fluid.layers.transpose(x=atten_fc3, perm=[0, 2, 1])
    atten_fc3 = fluid.layers.scale(x=atten_fc3, scale=hidden_size ** -0.5)
    weight = fluid.layers.softmax(atten_fc3)
    out = fluid.layers.matmul(weight, hist)
    out = fluid.layers.reshape(x=out, shape=[0, hidden_size])
    return out


def bow_net(data, layer_name, dict_dim, flags, emb_dim=128, hid_dim=128,
        fc_dim=128, emb_lr=0.1):
    """
    bow net
    """
    # embedding layer
    emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
    
    # bow layer
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    #bow = fluid.layers.tanh(bow)
    #bow = fluid.layers.softsign(bow)
    
    # full connect layer
    if fc_dim > 0:
        bow = fluid.layers.fc(input=bow, size=fc_dim, act=flags.activate)
    return bow 


def cnn_net(data, layer_name, dict_dim, flags, emb_dim=128, hid_dim=128, fc_dim=96,
        win_size=3, emb_lr=0.1):
    """
    conv net
    """
    # embedding layer
    emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
    
    param_attr = fluid.ParamAttr(
        name="conv_weight",
        initializer=fluid.initializer.TruncatedNormalInitializer(loc=0.0, scale=0.1))
    bias_attr = fluid.ParamAttr(
        name="conv_bias",
        initializer=fluid.initializer.Constant(0.0))
    
    if flags.use_attention:
        convs = []
        win_sizes = [3]
        for idx, win_size in enumerate(win_sizes): 
            #param_attr = fluid.ParamAttr(
            #    name="conv_weight_%s" % idx,
            #    initializer=fluid.initializer.TruncatedNormalInitializer(loc=0.0, scale=0.1))
            #bias_attr = fluid.ParamAttr(
            #    name="conv_bias_%s" % idx,
            #    initializer=fluid.initializer.Constant(0.0))

            # convolution layer
            conv = fluid.layers.sequence_conv(
                input=emb,
                num_filters=hid_dim,
                filter_size=win_size,
                param_attr=param_attr,
                bias_attr=bias_attr,
                act="leaky_relu") #tanh
            convs.append(conv)
        #convs_out = fluid.layers.concat(input=convs, axis=1)

        #if 'prefix' in data.name:
        #    fluid.layers.Print(emb, summarize=10000)
        #    fluid.layers.Print(conv, summarize=10000)
        #TODO: 3-layer cnn
        pad_value = fluid.layers.fill_constant(shape=[1], value=0.0, dtype='float32')
        conv, lens = fluid.layers.sequence_pad(conv, pad_value) #B, S, H
        mask = fluid.layers.cast(fluid.layers.sequence_mask(lens), "float32")
        mask = fluid.layers.unsqueeze(mask, axes=[2])
        att = dot_product_attention(conv, conv, conv, hid_dim, mask, mask, flags.dropout, name=data.name)
        #add residual layer
        conv = [conv, att, lens]
    else:
        # convolution layer
        conv = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=win_size,
            param_attr=param_attr,
            bias_attr=bias_attr,
            act="leaky_relu", #tanh 
            pool_type="max")
        # full connect layer
        if fc_dim > 0:
            conv = fluid.layers.fc(input=conv, size=fc_dim, act=flags.activate)
    return conv


def lstm_net(data, layer_name, dict_dim, flags, emb_dim=128, hid_dim=128,
        fc_dim=96, emb_lr=0.1):
    """
    lstm net
    """
    # embedding layer
    emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
    
    # Lstm layer
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4,
            param_attr=fluid.ParamAttr(name='lstm_fc_weight'),
            bias_attr=fluid.ParamAttr(name='lstm_fc_bias'))
    lstm_h, c = fluid.layers.dynamic_lstm(input=fc0, size=hid_dim * 4, is_reverse=False,
            param_attr=fluid.ParamAttr(name='lstm_weight'),
            bias_attr=fluid.ParamAttr(name='lstm_bias'))
    # max pooling layer
    lstm = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm = fluid.layers.tanh(lstm)

    # full connect layer
    if fc_dim > 0:
        lstm = fluid.layers.fc(input=lstm, size=fc_dim, act=flags.activate)
    return lstm


def bilstm_net(data, layer_name, dict_dim, flags, emb_dim=128, hid_dim=128,
        fc_dim=96, emb_lr=0.1):
    """
    bi-lstm net
    """
    # embedding layer
    emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
    
    #LSTM layer
    ffc = fluid.layers.fc(input=emb, size=hid_dim * 4)
    rfc = fluid.layers.fc(input=emb, size=hid_dim * 4)
    flstm_h, _ = fluid.layers.dynamic_lstm(input=ffc, size=hid_dim * 4, is_reverse=False)
    rlstm_h, _ = fluid.layers.dynamic_lstm(input=rfc, size=hid_dim * 4, is_reverse=True)
   
    if flags.use_attention:
        lstm_concat = fluid.layers.concat(
            input=[flstm_h, rlstm_h], axis=1)
        #fluid.layers.Print(lstm_concat) 
        bi_lstm = general_attention(lstm_concat, flags.dropout)
    else:
        # extract last layer
        flstm_last = fluid.layers.sequence_last_step(input=flstm_h)
        rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)
        #flstm_last = fluid.layers.tanh(flstm_last)
        #rlstm_last = fluid.layers.tanh(rlstm_last)
        # concat layer
        bi_lstm = fluid.layers.concat(input=[flstm_last, rlstm_last], axis=1)
        
        # full connect layer
        if fc_dim > 0:
            bi_lstm = fluid.layers.fc(input=bi_lstm, size=fc_dim, act=flags.activate)
    return bi_lstm 
 

def gru_net(data, layer_name, dict_dim, flags, emb_dim=128, hid_dim=128,
        fc_dim=96, emb_lr=0.1):
    """
    gru net
    """
    emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
   
    #gru layer
    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 3)
    gru = fluid.layers.dynamic_gru(input=fc0, size=hid_dim, is_reverse=False)
    gru = fluid.layers.sequence_pool(input=gru, pool_type='max')
    #gru = fluid.layers.tanh(gru)
    
    if fc_dim > 0:
        gru = fluid.layers.fc(input=gru, size=fc_dim, act=flags.activate)
    return gru


def textcnn_net(data, layer_name, dict_dim, flags, emb_dim=128, hid_dim=128,
        fc_dim=96, win_sizes=None, emb_lr=0.1):
    """
    textcnn_net
    """
    if win_sizes is None:
        win_sizes = [1, 2, 3]

    # embedding layer
    emb = fluid.layers.embedding(input=data, is_sparse=True, size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)

    # convolution layer
    convs = []
    for win_size in win_sizes: 
        param_attr = fluid.ParamAttr(
            name="conv_%s" % win_size,
            initializer=fluid.initializer.TruncatedNormalInitializer(loc=0.0, scale=0.1))
        bias_attr = fluid.ParamAttr(
            name="bias_%s" % win_size,
            initializer=fluid.initializer.Constant(0.0))

        conv_h = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=win_size,
            param_attr=param_attr,
            bias_attr=bias_attr,
            act="tanh",
            pool_type="max")
        convs.append(conv_h)
    convs_out = fluid.layers.concat(input=convs, axis=1)

    # full connect layer
    if fc_dim > 0:
        convs_out = fluid.layers.fc(input=[convs_out], size=fc_dim, act=flags.activate)
    return convs_out
 
 
def dp_cnn(data, layer_name, dict_dim, flags, emb_dim=128, hid_dim=128,
        fc_dim=96, emb_lr=0.1):
    "deep cnn"

    channel_size = 250
    blocks = 6

    def _block(x):
        x = fluid.layers.relu(x)
        x = fluid.layers.conv2d(x, channel_size, (3, 1), padding=(1, 0))
        x = fluid.layers.relu(x)
        x = fluid.layers.conv2d(x, channel_size, (3, 1), padding=(1, 0))
        return x

    # embedding layer
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(name=layer_name, learning_rate=emb_lr), padding_idx=0)
    emb = fluid.layers.unsqueeze(emb, axes=[1])
    region_embedding = fluid.layers.conv2d(emb, channel_size, (3, emb_dim), padding=(1, 0))
    conv_features = _block(region_embedding)
    conv_features = conv_features + region_embedding
    for i in range(blocks):
        block_features = fluid.layers.pool2d(conv_features, 
                            pool_size=(3, 1), 
                            pool_stride=(2, 1), 
                            pool_padding=(1, 0))
        conv_features = _block(block_features)
        conv_features = block_features + conv_features
    features = fluid.layers.pool2d(conv_features, global_pooling=True)
    features = fluid.layers.squeeze(features, axes=[2, 3])
    # full connect layer
    if fc_dim > 0:
        features = fluid.layers.fc(input=[features], size=fc_dim, act=flags.activate)

    return features

