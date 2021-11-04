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
"""Transformer decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.attention import multi_head_attention, multi_head_hierarchical_attention
from models.neural_modules import positionwise_feed_forward, \
    pre_process_layer, post_process_layer


def transformer_decoder_layer(dec_input,
                              enc_output,
                              slf_attn_bias,
                              dec_enc_attn_bias,
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
                              cache=None,
                              gather_idx=None,
                              param_initializer=None,
                              name=''):
    """
    The layer to be stacked in decoder part.
    :param dec_input:  (batch_size, tgt_len, emb_dim)
    :param enc_output:  (batch_size, n_tokens, emb_dim)
    :param slf_attn_bias:  (batch_size, n_head, tgt_len, tgt_len)
    :param dec_enc_attn_bias:  (batch_size, n_head, tgt_len, n_tokens)
    """
    # (batch_size, tgt_len, emb_dim)
    slf_attn_output = multi_head_attention(
        queries=pre_process_layer(out=dec_input,  # add layer normalization
                                  process_cmd=preprocess_cmd,
                                  dropout_rate=prepostprocess_dropout,
                                  name=name + '_pre_slf_attn'),
        keys=None,
        values=None,
        attn_bias=slf_attn_bias,  # (batch_size, n_head, tgt_len, tgt_len)
        d_key=d_key,
        d_value=d_value,
        d_model=d_model,
        n_head=n_head,
        dropout_rate=attention_dropout,
        cache=cache,
        gather_idx=gather_idx,
        param_initializer=param_initializer,
        name=name + '_slf_attn')

    # add dropout and residual connection
    # (batch_size, tgt_len, emb_dim)
    slf_attn_output = post_process_layer(
        prev_out=dec_input,
        out=slf_attn_output,
        process_cmd=postprocess_cmd,
        dropout_rate=prepostprocess_dropout,
        name=name + '_post_slf_attn')

    # (batch_size, tgt_len, emb_dim)
    context_attn_output = multi_head_attention(
        queries=pre_process_layer(out=slf_attn_output,  # add layer normalization
                                  process_cmd=preprocess_cmd,
                                  dropout_rate=prepostprocess_dropout,
                                  name=name + '_pre_context_attn'),
        keys=enc_output,  # (batch_size, n_tokens, emb_dim)
        values=enc_output,  # (batch_size, n_tokens, emb_dim)
        attn_bias=dec_enc_attn_bias,  # (batch_size, n_head, tgt_len, n_tokens)
        d_key=d_key,
        d_value=d_value,
        d_model=d_model,
        n_head=n_head,
        dropout_rate=attention_dropout,
        cache=cache,
        gather_idx=gather_idx,
        static_kv=True,
        param_initializer=param_initializer,
        name=name + '_context_attn')

    # add dropout and residual connection
    context_attn_output = post_process_layer(
        prev_out=slf_attn_output,
        out=context_attn_output,
        process_cmd=postprocess_cmd,
        dropout_rate=prepostprocess_dropout,
        name=name + '_post_context_attn')

    ffd_output = positionwise_feed_forward(
        x=pre_process_layer(out=context_attn_output,  # add layer normalization
                            process_cmd=preprocess_cmd,
                            dropout_rate=prepostprocess_dropout,
                            name=name + '_pre_ffn'),
        d_inner_hid=d_inner_hid,
        d_hid=d_model,
        dropout_rate=relu_dropout,
        hidden_act=hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn')

    # add dropout and residual connection
    dec_output = post_process_layer(
        prev_out=context_attn_output,
        out=ffd_output,
        process_cmd=postprocess_cmd,
        dropout_rate=prepostprocess_dropout,
        name=name + '_post_ffn')

    return dec_output  # (batch_size, tgt_len, emb_dim)


def transformer_decoder(dec_input,
                        enc_output,
                        dec_slf_attn_bias,
                        dec_enc_attn_bias,
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
                        preprocess_cmd,
                        postprocess_cmd,
                        caches=None,
                        gather_idx=None,
                        param_initializer=None,
                        name='transformer_decoder'):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    :param dec_input:  (batch_size, tgt_len, emb_dim)
    :param enc_output:  (batch_size, n_tokens, emb_dim)
    :param dec_slf_attn_bias:  (batch_size, n_head, tgt_len, tgt_len)
    :param dec_enc_attn_bias:  (batch_size, n_head, tgt_len, n_tokens)
    """
    for i in range(n_layer):
        # (batch_size, tgt_len, emb_dim)
        dec_output = transformer_decoder_layer(
            dec_input=dec_input,
            enc_output=enc_output,
            slf_attn_bias=dec_slf_attn_bias,
            dec_enc_attn_bias=dec_enc_attn_bias,
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
            cache=None if caches is None else caches[i],
            gather_idx=gather_idx,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i))

        dec_input = dec_output

    # add layer normalization
    dec_output = pre_process_layer(out=dec_output,
                                   process_cmd=preprocess_cmd,
                                   dropout_rate=prepostprocess_dropout,
                                   name=name + '_post')

    return dec_output  # (batch_size, tgt_len, emb_dim)


def graph_decoder_layer(dec_input,
                        enc_words_output,
                        enc_sents_output,
                        slf_attn_bias,
                        dec_enc_words_attn_bias,
                        dec_enc_sents_attn_bias,
                        graph_attn_bias,
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
                        preprocess_cmd,
                        postprocess_cmd,
                        cache=None,
                        gather_idx=None,
                        param_initializer=None,
                        name=''):
    """
    The layer to be stacked in decoder part.
    :param dec_input:  (batch_size, tgt_len, emb_dim)
    :param enc_words_output:  (batch_size, n_blocks, n_tokens, emb_dim)
    :param enc_sents_output:  (batch_size, n_blocks, emb_dim)
    :param slf_attn_bias:  (batch_size, n_head, tgt_len, tgt_len)
    :param dec_enc_words_attn_bias:  (batch_size, n_blocks, n_head, tgt_len, n_tokens)
    :param dec_enc_sents_attn_bias:  (batch_size, n_head, tgt_len, n_blocks)
    :param graph_attn_bias:  (batch_size, n_head, n_blocks, n_blocks)
    """
    # (batch_size, tgt_len, emb_dim)
    slf_attn_output = multi_head_attention(
        queries=pre_process_layer(out=dec_input,  # add layer normalization
                                  process_cmd=preprocess_cmd,
                                  dropout_rate=prepostprocess_dropout,
                                  name=name + '_pre_attn'),
        keys=None,
        values=None,
        attn_bias=slf_attn_bias,  # (batch_size, n_head, tgt_len, tgt_len)
        d_key=d_key,
        d_value=d_value,
        d_model=d_model,
        n_head=n_head,
        dropout_rate=attention_dropout,
        cache=cache,
        gather_idx=gather_idx,
        name=name + '_attn')

    # add dropout and residual connection
    # (batch_size, tgt_len, emb_dim)
    slf_attn_output = post_process_layer(
        prev_out=dec_input,
        out=slf_attn_output,
        process_cmd=postprocess_cmd,
        dropout_rate=prepostprocess_dropout,
        name=name + '_post_attn'
    )

    # (batch_size, tgt_len, emb_dim)
    hier_attn_output = multi_head_hierarchical_attention(
        queries=pre_process_layer(out=slf_attn_output,  # add layer normalization
                                  process_cmd=preprocess_cmd,
                                  dropout_rate=prepostprocess_dropout,
                                  name=name + '_pre_hier_attn'),
        keys_w=enc_words_output,  # (batch_size, n_blocks, n_tokens, emb_dim)
        values_w=enc_words_output,  # (batch_size, n_blocks, n_tokens, emb_dim)
        attn_bias_w=dec_enc_words_attn_bias,  # (batch_size, n_blocks, n_head, tgt_len, n_tokens)
        keys_s=enc_sents_output,  # (batch_size, n_blocks, emb_dim)
        values_s=enc_sents_output,  # (batch_size, n_blocks, emb_dim)
        attn_bias_s=dec_enc_sents_attn_bias,  # (batch_size, n_head, tgt_len, n_blocks)
        graph_attn_bias=graph_attn_bias,  # (batch_size, n_head, n_blocks, n_blocks)
        pos_win=pos_win,
        d_key=d_key,
        d_value=d_value,
        d_model=d_model,
        n_head=n_head,
        dropout_rate=attention_dropout,
        cache=cache,
        gather_idx=gather_idx,
        name=name + '_hier_attn')

    # add dropout and residual connection
    hier_attn_output = post_process_layer(
        prev_out=slf_attn_output,
        out=hier_attn_output,
        process_cmd=postprocess_cmd,
        dropout_rate=prepostprocess_dropout,
        name=name + '_post_hier_attn')

    ffd_output = positionwise_feed_forward(
        x=pre_process_layer(out=hier_attn_output,  # add layer normalization
                            process_cmd=preprocess_cmd,
                            dropout_rate=prepostprocess_dropout,
                            name=name + '_pre_ffn'),
        d_inner_hid=d_inner_hid,
        d_hid=d_model,
        dropout_rate=relu_dropout,
        hidden_act=hidden_act,
        param_initializer=param_initializer,
        name=name + '_ffn'
    )

    # add dropout and residual connection
    dec_output = post_process_layer(
        prev_out=hier_attn_output,
        out=ffd_output,
        process_cmd=postprocess_cmd,
        dropout_rate=prepostprocess_dropout,
        name=name + '_post_ffn'
    )

    return dec_output  # (batch_size, tgt_len, emb_dim)


def graph_decoder(dec_input,
                  enc_words_output,
                  enc_sents_output,
                  dec_slf_attn_bias,
                  dec_enc_words_attn_bias,
                  dec_enc_sents_attn_bias,
                  graph_attn_bias,
                  pos_win,
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
                  preprocess_cmd,
                  postprocess_cmd,
                  caches=None,
                  gather_idx=None,
                  param_initializer=None,
                  name='graph_decoder'):
    """
    The decoder is composed of a stack of identical decoder_layer layers.
    :param dec_input:  (batch_size, tgt_len, emb_dim)
    :param enc_words_output:  (batch_size, n_blocks, n_tokens, emb_dim)
    :param enc_sents_output:  (batch_size, n_blocks, emb_dim)
    :param dec_slf_attn_bias:  (batch_size, n_head, tgt_len, tgt_len)
    :param dec_enc_words_attn_bias:  (batch_size, n_blocks, n_head, tgt_len, n_tokens)
    :param dec_enc_sents_attn_bias:  (batch_size, n_head, tgt_len, n_blocks)
    :param graph_attn_bias:  (batch_size, n_head, n_blocks, n_blocks)
    :return:
    """
    for i in range(n_layer):
        # (batch_size, tgt_len, emb_dim)
        dec_output = graph_decoder_layer(
            dec_input=dec_input,
            enc_words_output=enc_words_output,
            enc_sents_output=enc_sents_output,
            slf_attn_bias=dec_slf_attn_bias,
            dec_enc_words_attn_bias=dec_enc_words_attn_bias,
            dec_enc_sents_attn_bias=dec_enc_sents_attn_bias,
            graph_attn_bias=graph_attn_bias,
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
            cache=None if caches is None else caches[i],
            gather_idx=gather_idx,
            param_initializer=param_initializer,
            name=name + '_layer_' + str(i)
        )

        dec_input = dec_output

    # add layer normalization
    dec_output = pre_process_layer(out=dec_output,
                                   process_cmd=preprocess_cmd,
                                   dropout_rate=prepostprocess_dropout,
                                   name=name + '_post')
    return dec_output  # (batch_size, tgt_len, emb_dim)
