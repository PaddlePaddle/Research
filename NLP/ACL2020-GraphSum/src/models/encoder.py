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
"""Transformer encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers

from models.neural_modules import positionwise_feed_forward, \
    pre_process_layer, post_process_layer
from models.attention import multi_head_attention, multi_head_pooling, \
    multi_head_structure_attention


def transformer_encoder_layer(query_input,
                              key_input,
                              attn_bias,
                              n_head,
                              d_key,
                              d_value,
                              d_model,
                              d_inner_hid,
                              prepostprocess_dropout,
                              attention_dropout,
                              relu_dropout,
                              hidden_act,
                              preprocess_cmd="n",
                              postprocess_cmd="da",
                              param_initializer=None,
                              name=''):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    key_input = pre_process_layer(
        key_input,
        preprocess_cmd,
        prepostprocess_dropout,
        name=name + '_pre_att') if key_input else None
    value_input = key_input if key_input else None

    attn_output = multi_head_attention(
        pre_process_layer(
            query_input,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_att'),
        key_input,
        value_input,
        attn_bias,
        d_key,
        d_value,
        d_model,
        n_head,
        attention_dropout,
        param_initializer=param_initializer,
        name=name + '_multi_head_att')
    attn_output = post_process_layer(
        query_input,
        attn_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_att')
    ffd_output = positionwise_feed_forward(
        pre_process_layer(
            attn_output,
            preprocess_cmd,
            prepostprocess_dropout,
            name=name + '_pre_ffn'),
        d_inner_hid,
        d_model,
        relu_dropout,
        hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')
    return post_process_layer(
        attn_output,
        ffd_output,
        postprocess_cmd,
        prepostprocess_dropout,
        name=name + '_post_ffn')


def transformer_encoder(enc_input,
                        attn_bias,
                        n_layer,
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        hidden_act,
                        preprocess_cmd="n",
                        postprocess_cmd="da",
                        param_initializer=None,
                        name='transformer_encoder',
                        with_post_process=True):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    for i in range(n_layer):
        enc_output = transformer_encoder_layer(
            enc_input,
            None,
            attn_bias,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd,
            postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i))
        enc_input = enc_output

    if with_post_process:
        enc_output = pre_process_layer(
            enc_output, preprocess_cmd, prepostprocess_dropout, name="post_encoder")

    return enc_output


def self_attention_pooling_layer(enc_input,
                                 attn_bias,
                                 n_head,
                                 d_key,
                                 d_value,
                                 d_model,
                                 d_inner_hid,
                                 prepostprocess_dropout,
                                 attention_dropout,
                                 relu_dropout,
                                 n_block,
                                 preprocess_cmd="n",
                                 postprocess_cmd="da",
                                 name='self_attention_pooling'):
    """
    enc_input: # (batch_size*n_blocks, n_tokens, emb_dim)
    attn_bias:  # (batch_size*n_blocks, n_head, n_tokens, n_tokens)
    """
    attn_output = multi_head_pooling(
        keys=pre_process_layer(enc_input,
                               preprocess_cmd,
                               prepostprocess_dropout,
                               name=name + '_pre'),  # add layer normalization
        values=None,
        attn_bias=attn_bias,  # (batch_size*n_blocks, n_head, n_tokens, n_tokens)
        d_value=d_value,
        d_model=d_model,
        n_head=n_head,
        dropout_rate=attention_dropout,
        name=name
    )  # (batch_size*n_blocks, d_model)

    # print("n_block = %s" % n_block)
    # print("attn_output.shape = %s" % str(attn_output.shape))
    attn_output = layers.reshape(attn_output, shape=[-1, n_block, d_model])
    # print("attn_output.shape = %s" % str(attn_output.shape))

    pooling_output = layers.dropout(
        attn_output,
        dropout_prob=attention_dropout,
        dropout_implementation="upscale_in_train",
        is_test=False)

    return pooling_output


def graph_encoder_layer(enc_input,  # (batch_size, n_block, emb_dim)
                        attn_bias,  # (batch_size, n_head, n_block, n_block)
                        graph_attn_bias,  # (batch_size, n_head, n_block, n_block)
                        pos_win,
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        hidden_act,
                        preprocess_cmd="n",
                        postprocess_cmd="da",
                        param_initializer=None,
                        name=''):
    """
    :param enc_input:  (batch_size, n_blocks, emb_dim)
    :param attn_bias:  (batch_size, n_head, n_blocks, n_blocks)
    :param graph_attn_bias:  (batch_size, n_head, n_blocks, n_blocks)
    """
    # (batch_size, n_block, d_model)
    attn_output = multi_head_structure_attention(
        queries=pre_process_layer(out=enc_input,  # add layer normalization
                                  process_cmd=preprocess_cmd,
                                  dropout_rate=prepostprocess_dropout,
                                  name=name + '_pre_attn'),
        keys=None,
        values=None,
        attn_bias=attn_bias,
        graph_attn_bias=graph_attn_bias,
        pos_win=pos_win,
        d_key=d_key,
        d_value=d_value,
        d_model=d_model,
        n_head=n_head,
        dropout_rate=attention_dropout,
        name=name + '_graph_attn'
    )

    # add dropout and residual connection
    attn_output = post_process_layer(prev_out=enc_input,
                                     out=attn_output,
                                     process_cmd=postprocess_cmd,
                                     dropout_rate=prepostprocess_dropout,
                                     name=name + '_post_attn')

    ffd_output = positionwise_feed_forward(
        x=pre_process_layer(out=attn_output,  # add layer normalization
                            process_cmd=preprocess_cmd,
                            dropout_rate=prepostprocess_dropout,
                            name=name + '_pre_ffn'),
        d_inner_hid=d_inner_hid,
        d_hid=d_model,
        dropout_rate=relu_dropout,
        hidden_act=hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')

    return post_process_layer(prev_out=attn_output,  # add dropout and residual connection
                              out=ffd_output,
                              process_cmd=postprocess_cmd,
                              dropout_rate=prepostprocess_dropout,
                              name=name + '_post_ffn')


def graph_encoder(enc_words_output,
                  src_words_slf_attn_bias,
                  src_sents_slf_attn_bias,
                  graph_attn_bias,
                  pos_win,
                  graph_layers,
                  n_head,
                  d_key,
                  d_value,
                  d_model,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  param_initializer=None,
                  name='graph_encoder'):
    """
    :param enc_words_output:  # (batch_size*n_blocks, n_tokens, emb_dim)
    :param src_words_slf_attn_bias:  (batch_size*n_block, n_head, n_tokens, n_tokens)
    :param src_sents_slf_attn_bias:  (batch_size, n_head, n_block, n_block)
    :param graph_attn_bias:  (batch_size, n_head, n_block, n_block)
    :return:
    """
    # (batch_size, n_block, d_model)
    sents_vec = self_attention_pooling_layer(enc_input=enc_words_output,
                                             attn_bias=src_words_slf_attn_bias,
                                             n_head=n_head,
                                             d_key=d_key,
                                             d_value=d_value,
                                             d_model=d_model,
                                             d_inner_hid=d_inner_hid,
                                             prepostprocess_dropout=prepostprocess_dropout,
                                             attention_dropout=attention_dropout,
                                             relu_dropout=relu_dropout,
                                             n_block=src_sents_slf_attn_bias.shape[2],
                                             preprocess_cmd="n",
                                             postprocess_cmd="da",
                                             name=name + '_pooling')

    enc_input = sents_vec  # (batch_size, n_block, d_model)

    for i in range(graph_layers):
        # (batch_size, n_block, emb_dim)
        enc_output = graph_encoder_layer(
            enc_input=enc_input,  # (batch_size, n_block, emb_dim)
            attn_bias=src_sents_slf_attn_bias,  # (batch_size, n_head, n_block, n_block)
            graph_attn_bias=graph_attn_bias,  # (batch_size, n_head, n_block, n_block)
            pos_win=pos_win,
            n_head=n_head,
            d_key=d_key,
            d_value=d_value,
            d_model=d_model,
            d_inner_hid=d_inner_hid,
            prepostprocess_dropout=prepostprocess_dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            hidden_act=hidden_act,
            preprocess_cmd=preprocess_cmd,
            postprocess_cmd=postprocess_cmd,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i)
        )

        enc_input = enc_output  # (batch_size, n_block, emb_dim)

    # add layer normalization
    enc_output = pre_process_layer(out=enc_output,
                                   process_cmd=preprocess_cmd,
                                   dropout_rate=prepostprocess_dropout,
                                   name=name + '_post')
    return enc_output  # (batch_size, n_block, emb_dim)
