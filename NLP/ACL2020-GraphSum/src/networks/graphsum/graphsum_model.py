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
"""GraphSum model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
from collections import namedtuple
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from models.encoder import transformer_encoder, graph_encoder
from models.decoder import graph_decoder
from models.neural_modules import pre_process_layer
from models.trigram_blocking import TrigramBlocking

INF = 1. * 1e18


class GraphSumConfig(object):
    """Parser for configuration files"""

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        """Print configuration"""
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class GraphSumModel(object):
    """GraphSum Model"""

    def __init__(self, args, config, padding_idx, bos_idx, eos_idx, tokenizer):
        self.args = args
        self._emb_size = config['hidden_size']
        self._enc_word_layer = config['enc_word_layers']
        self._enc_graph_layer = config['enc_graph_layers']
        self._dec_n_layer = config['dec_graph_layers']
        self._n_head = config['num_attention_heads']
        self._max_position_seq_len = config['max_position_embeddings']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._preprocess_command = config['preprocess_command']
        self._postprocess_command = config['postprocess_command']
        self._word_emb_name = config['word_embedding_name']
        self._enc_word_pos_emb_name = config['enc_word_pos_embedding_name']
        self._enc_sen_pos_emb_name = config['enc_sen_pos_embedding_name']
        self._dec_word_pos_emb_name = config['dec_word_pos_embedding_name']

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._label_smooth_eps = args.label_smooth_eps
        self._padding_idx = padding_idx
        self._weight_sharing = args.weight_sharing
        self._dtype = "float16" if args.use_fp16 else "float32"
        self._use_fp16 = args.use_fp16
        self._emb_dtype = "float32"
        self.beam_size = args.beam_size
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.tokenizer = tokenizer  # spm tokenizer
        self.voc_size = len(tokenizer)
        self.max_para_len = args.max_para_len
        self.max_para_num = args.max_para_num
        self.graph_type = args.graph_type
        self.max_tgt_len = args.max_tgt_len
        self.len_penalty = args.len_penalty
        self.max_out_len = args.max_out_len
        self.min_out_len = args.min_out_len
        self.block_trigram = args.block_trigram
        self.pos_win = args.pos_win

    def _gen_enc_input(self, src_word, src_word_pos, src_sen_pos, word_slf_attn_bias,
                       sen_slf_attn_bias, graph_attn_bias):
        # (batch_size, max_n_block, max_n_token, emb_dim)
        word_emb_out = fluid.layers.embedding(
            input=src_word,
            size=[self.voc_size, self._emb_size],
            padding_idx=self._padding_idx,  # set embedding of pad to 0
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        word_emb_out = layers.scale(x=word_emb_out, scale=self._emb_size ** 0.5)

        # (batch_size, max_n_block, max_n_token, emb_dim/2)
        word_pos_out = fluid.layers.embedding(
            input=src_word_pos,
            size=[self._max_position_seq_len, self._emb_size // 2],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._enc_word_pos_emb_name, trainable=False))
        word_pos_out.stop_gradient = True

        # (batch_size, max_n_block, emb_dim/2)
        sen_pos_out = fluid.layers.embedding(
            input=src_sen_pos,
            size=[self._max_position_seq_len, self._emb_size // 2],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._enc_sen_pos_emb_name, trainable=False))
        sen_pos_out.stop_gradient = True

        # (batch_size, max_n_block, max_n_token, emb_dim/2)
        sen_pos_out = layers.expand(layers.unsqueeze(sen_pos_out, axes=[2]),
                                    expand_times=[1, 1, self.max_para_len, 1])

        # (batch_size, n_blocks, n_tokens, emb_dim)
        combined_pos_enc = layers.concat([word_pos_out, sen_pos_out], axis=-1)

        emb_out = word_emb_out + combined_pos_enc  # (batch_size, n_blocks, n_tokens, emb_dim)

        emb_out = layers.dropout(emb_out,
                                 dropout_prob=self._prepostprocess_dropout,
                                 dropout_implementation="upscale_in_train",
                                 is_test=False) if self._prepostprocess_dropout else emb_out

        if self._dtype is "float16":
            emb_out = fluid.layers.cast(x=emb_out, dtype=self._dtype)

            if word_slf_attn_bias is not None:
                word_slf_attn_bias = fluid.layers.cast(x=word_slf_attn_bias, dtype=self._dtype)

            if sen_slf_attn_bias is not None:
                sen_slf_attn_bias = fluid.layers.cast(x=sen_slf_attn_bias, dtype=self._dtype)

            if graph_attn_bias is not None:
                graph_attn_bias = fluid.layers.cast(x=graph_attn_bias, dtype=self._dtype)

        res = namedtuple('results', ['emb_out', 'word_slf_attn_bias', 'sen_slf_attn_bias',
                                     'graph_attn_bias'])

        return res(emb_out=emb_out, word_slf_attn_bias=word_slf_attn_bias,
                   sen_slf_attn_bias=sen_slf_attn_bias, graph_attn_bias=graph_attn_bias)

    def _gen_dec_input(self, trg_word, trg_pos, trg_slf_attn_bias, trg_src_words_attn_bias,
                       trg_src_sents_attn_bias, graph_attn_bias):
        emb_out = fluid.layers.embedding(
            input=trg_word,
            size=[self.voc_size, self._emb_size],
            padding_idx=self._padding_idx,  # set embedding of pad to 0
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        emb_out = layers.scale(x=emb_out, scale=self._emb_size ** 0.5)

        position_emb_out = fluid.layers.embedding(
            input=trg_pos,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(
                name=self._dec_word_pos_emb_name, trainable=False))
        position_emb_out.stop_gradient = True

        emb_out = emb_out + position_emb_out
        emb_out = layers.dropout(
            emb_out,
            dropout_prob=self._prepostprocess_dropout,
            dropout_implementation="upscale_in_train",
            is_test=False) if self._prepostprocess_dropout else emb_out

        if self._dtype is "float16":
            emb_out = fluid.layers.cast(x=emb_out, dtype=self._dtype)
            if trg_slf_attn_bias is not None:
                trg_slf_attn_bias = fluid.layers.cast(x=trg_slf_attn_bias, dtype=self._dtype)

            if trg_src_words_attn_bias is not None:
                trg_src_words_attn_bias = fluid.layers.cast(x=trg_src_words_attn_bias, dtype=self._dtype)

            if trg_src_sents_attn_bias is not None:
                trg_src_sents_attn_bias = fluid.layers.cast(x=trg_src_sents_attn_bias, dtype=self._dtype)

            if graph_attn_bias is not None:
                graph_attn_bias = fluid.layers.cast(x=graph_attn_bias, dtype=self._dtype)

        res = namedtuple('results', ['emb_out', 'trg_slf_attn_bias', 'trg_src_words_attn_bias',
                                     'trg_src_sents_attn_bias', 'graph_attn_bias'])

        return res(emb_out=emb_out, trg_slf_attn_bias=trg_slf_attn_bias,
                   trg_src_words_attn_bias=trg_src_words_attn_bias,
                   trg_src_sents_attn_bias=trg_src_sents_attn_bias,
                   graph_attn_bias=graph_attn_bias)

    def encode(self, enc_input):
        """Encoding the source input"""

        src_word, src_word_pos, src_sen_pos, src_words_slf_attn_bias, \
        src_sents_slf_attn_bias, graph_attn_bias = enc_input

        enc_res = self._gen_enc_input(src_word, src_word_pos, src_sen_pos, src_words_slf_attn_bias,
                                      src_sents_slf_attn_bias, graph_attn_bias)

        emb_out, src_words_slf_attn_bias, src_sents_slf_attn_bias, graph_attn_bias = \
            enc_res.emb_out, enc_res.word_slf_attn_bias, enc_res.sen_slf_attn_bias, enc_res.graph_attn_bias

        # (batch_size*n_blocks, n_tokens, emb_dim)
        emb_out = layers.reshape(emb_out, shape=[-1, self.max_para_len, self._emb_size])

        # (batch_size*n_block, n_head, n_tokens, n_tokens)
        src_words_slf_attn_bias = layers.reshape(src_words_slf_attn_bias,
                                                 shape=[-1, self._n_head, self.max_para_len, self.max_para_len])

        # the token-level transformer encoder
        # (batch_size*n_blocks, n_tokens, emb_dim)
        enc_words_out = transformer_encoder(
            enc_input=emb_out,
            attn_bias=src_words_slf_attn_bias,
            n_layer=self._enc_word_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=self._prepostprocess_dropout,
            hidden_act=self._hidden_act,
            preprocess_cmd=self._preprocess_command,
            postprocess_cmd=self._postprocess_command,
            param_initializer=self._param_initializer,
            name='transformer_encoder',
            with_post_process=False
        )

        # the paragraph-level graph encoder
        # (batch_size, n_block, emb_dim)
        enc_sents_out = graph_encoder(
            enc_words_output=enc_words_out,  # (batch_size*n_blocks, n_tokens, emb_dim)
            src_words_slf_attn_bias=src_words_slf_attn_bias,  # (batch_size*max_nblock, n_head, max_ntoken, max_ntoken)
            src_sents_slf_attn_bias=src_sents_slf_attn_bias,  # (batch_size, n_head, max_nblock, max_nblock)
            graph_attn_bias=graph_attn_bias,  # (batch_size, n_head, max_nblock, max_nblock)
            pos_win=self.pos_win,
            graph_layers=self._enc_graph_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=self._prepostprocess_dropout,
            hidden_act=self._hidden_act,
            # n_block=self.max_para_num,
            preprocess_cmd=self._preprocess_command,
            postprocess_cmd=self._postprocess_command,
            param_initializer=self._param_initializer,
            name='graph_encoder')

        enc_words_out = pre_process_layer(
            enc_words_out, self._preprocess_command, self._prepostprocess_dropout, name="post_encoder")

        enc_words_out = layers.reshape(enc_words_out,
                                       shape=[-1, self.max_para_num, self.max_para_len, self._emb_size])

        return enc_words_out, enc_sents_out

    def decode(self, dec_input, enc_words_output, enc_sents_output, caches=None, gather_idx=None):
        """Decoding to generate output text"""

        trg_word, trg_pos, trg_slf_attn_bias, trg_src_words_attn_bias, \
        trg_src_sents_attn_bias, graph_attn_bias = dec_input

        dec_res = self._gen_dec_input(trg_word, trg_pos, trg_slf_attn_bias, trg_src_words_attn_bias,
                                      trg_src_sents_attn_bias, graph_attn_bias)

        emb_out, trg_slf_attn_bias, trg_src_words_attn_bias, trg_src_sents_attn_bias, graph_attn_bias = \
            dec_res.emb_out, dec_res.trg_slf_attn_bias, dec_res.trg_src_words_attn_bias, \
            dec_res.trg_src_sents_attn_bias, dec_res.graph_attn_bias

        # (batch_size, tgt_len, emb_dim)
        dec_output = graph_decoder(
            dec_input=emb_out,  # (batch_size, tgt_len, emb_dim)
            enc_words_output=enc_words_output,  # (batch_size, n_blocks, n_tokens, emb_dim)
            enc_sents_output=enc_sents_output,  # (batch_size, n_blocks, emb_dim)
            dec_slf_attn_bias=trg_slf_attn_bias,  # (batch_size, n_head, tgt_len, tgt_len)
            dec_enc_words_attn_bias=trg_src_words_attn_bias,  # (batch_size, n_blocks, n_head, tgt_len, n_tokens)
            dec_enc_sents_attn_bias=trg_src_sents_attn_bias,  # (batch_size, n_head, tgt_len, n_blocks)
            graph_attn_bias=graph_attn_bias,  # (batch_size, n_head, n_blocks, n_blocks)
            pos_win=self.pos_win,
            n_layer=self._dec_n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=self._prepostprocess_dropout,
            hidden_act=self._hidden_act,
            preprocess_cmd=self._preprocess_command,
            postprocess_cmd=self._postprocess_command,
            param_initializer=self._param_initializer,
            caches=caches,
            gather_idx=gather_idx,
            name='graph_decoder')

        # Reshape to 2D tensor to use GEMM instead of BatchedGEMM
        # (batch_size*tgt_len, emb_dim)
        dec_output = layers.reshape(dec_output, shape=[-1, self._emb_size], inplace=True)

        if self._dtype is "float16":
            dec_output = fluid.layers.cast(x=dec_output, dtype=self._emb_dtype)

        if self._weight_sharing:
            out = layers.matmul(
                x=dec_output,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            bias = layers.create_parameter(shape=[self.voc_size],
                                           dtype=self._emb_dtype,
                                           attr=fluid.ParamAttr(
                                               name='generator.bias',
                                               initializer=fluid.initializer.Constant(value=0.0)),
                                           is_bias=True)
            predict = layers.elementwise_add(x=out, y=bias, axis=-1)
        else:
            predict = layers.fc(input=dec_output,
                                size=self.voc_size,
                                param_attr=fluid.ParamAttr(
                                    name="generator.w",
                                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                                bias_attr=fluid.ParamAttr(
                                    name='generator.bias',
                                    initializer=fluid.initializer.Constant(value=0.0)))

        return predict

    def build_model(self, enc_input, dec_input, tgt_label, label_weights):
        """Build the model with source encoding and target decoding"""

        enc_word_output, enc_sen_output = self.encode(enc_input)
        dec_output = self.decode(dec_input, enc_word_output, enc_sen_output)

        predict_token_idx = layers.argmax(dec_output, axis=-1)
        correct_token_idx = layers.cast(layers.equal(tgt_label,
                                                     layers.reshape(predict_token_idx, shape=[-1, 1])),
                                        dtype='float32')
        weighted_correct = layers.elementwise_mul(x=correct_token_idx, y=label_weights, axis=0)
        sum_correct = layers.reduce_sum(weighted_correct)
        sum_correct.stop_gradient = True

        # Padding index do not contribute to the total loss. The weights is used to
        # cancel padding index in calculating the loss.
        if self._label_smooth_eps:
            # TODO: use fluid.input.one_hot after softmax_with_cross_entropy removing
            # the enforcement that the last dimension of label must be 1.
            tgt_label = layers.label_smooth(label=layers.one_hot(input=tgt_label,
                                                                 depth=self.voc_size),
                                            epsilon=self._label_smooth_eps)

        cost = layers.softmax_with_cross_entropy(
            logits=dec_output,
            label=tgt_label,
            soft_label=True if self._label_smooth_eps else False)

        weighted_cost = layers.elementwise_mul(x=cost, y=label_weights, axis=0)
        sum_cost = layers.reduce_sum(weighted_cost)
        token_num = layers.reduce_sum(label_weights)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num

        graph_vars = {
            "loss": avg_cost,
            "sum_correct": sum_correct,
            "token_num": token_num,
        }
        for k, v in graph_vars.items():
            v.persistable = True

        return graph_vars

    def create_model(self, pyreader_name, is_prediction=False):
        """Create the network"""

        if is_prediction:
            return self.fast_decode(pyreader_name)

        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, self.max_para_num, self.max_para_len],  # src_word
                    [-1, self.max_para_num, self.max_para_len],  # src_word_pos
                    [-1, self.max_para_num],  # src_sent_pos
                    [-1, self.max_para_num, self.max_para_len],  # src_words_slf_attn_bias
                    [-1, self.max_para_num],  # src_sents_slf_attn_bias
                    [-1, self.max_para_num, self.max_para_num],  # graph_attn_bias
                    [-1, self.max_tgt_len],  # trg_word
                    [-1, self.max_tgt_len],  # trg_pos
                    [-1, self.max_tgt_len, self.max_tgt_len],  # trg_slf_attn_bias
                    [-1, 1],  # tgt_label
                    [-1, 1]],  # label_weights
            dtypes=['int64', 'int64', 'int64', 'float32', 'float32', 'float32',
                    'int64', 'int64', 'float32',
                    'int64', 'float32'],
            lod_levels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)

        (src_word, src_word_pos, src_sent_pos, src_words_slf_attn_bias, src_sents_slf_attn_bias,
         graph_attn_bias, trg_word, trg_pos, trg_slf_attn_bias, tgt_label, label_weights) = \
            fluid.layers.read_file(pyreader)

        src_words_slf_attn_bias = layers.expand(layers.unsqueeze(src_words_slf_attn_bias, axes=[2, 3]),
                                                expand_times=[1, 1, self._n_head, self.max_para_len, 1])
        src_words_slf_attn_bias.stop_gradient = True
        src_sents_slf_attn_bias = layers.expand(layers.unsqueeze(src_sents_slf_attn_bias, axes=[1, 2]),
                                                expand_times=[1, self._n_head, self.max_para_num, 1])
        src_sents_slf_attn_bias.stop_gradient = True

        graph_attn_bias = layers.expand(layers.unsqueeze(graph_attn_bias, axes=[1]),
                                        expand_times=[1, self._n_head, 1, 1])
        graph_attn_bias.stop_gradient = True

        trg_slf_attn_bias = layers.expand(layers.unsqueeze(trg_slf_attn_bias, axes=[1]),
                                          expand_times=[1, self._n_head, 1, 1])
        trg_slf_attn_bias.stop_gradient = True

        tgt_src_words_attn_bias = layers.expand(layers.slice(src_words_slf_attn_bias, axes=[3],
                                                             starts=[0], ends=[1]),
                                                expand_times=[1, 1, 1, self.max_tgt_len, 1])
        tgt_src_words_attn_bias.stop_gradient = True

        tgt_src_sents_attn_bias = layers.expand(layers.slice(src_sents_slf_attn_bias, axes=[2],
                                                             starts=[0], ends=[1]),
                                                expand_times=[1, 1, self.max_tgt_len, 1])
        tgt_src_sents_attn_bias.stop_gradient = True

        src_word = layers.reshape(src_word, [-1, self.max_para_num, self.max_para_len, 1])
        src_word_pos = layers.reshape(src_word_pos, [-1, self.max_para_num, self.max_para_len, 1])
        src_sent_pos = layers.reshape(src_sent_pos, [-1, self.max_para_num, 1])
        trg_word = layers.reshape(trg_word, [-1, self.max_tgt_len, 1])
        trg_pos = layers.reshape(trg_pos, [-1, self.max_tgt_len, 1])
        tgt_label = layers.reshape(tgt_label, [-1, 1])
        label_weights = layers.reshape(label_weights, [-1, 1])

        enc_input = (src_word, src_word_pos, src_sent_pos, src_words_slf_attn_bias,
                     src_sents_slf_attn_bias, graph_attn_bias)
        dec_input = (trg_word, trg_pos, trg_slf_attn_bias, tgt_src_words_attn_bias,
                     tgt_src_sents_attn_bias, graph_attn_bias)

        graph_vars = self.build_model(enc_input=enc_input, dec_input=dec_input,
                                      tgt_label=tgt_label, label_weights=label_weights)

        return pyreader, graph_vars

    def fast_decode(self, pyreader_name):
        """Inference process of the model"""

        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, self.max_para_num, self.max_para_len],  # src_word
                    [-1, self.max_para_num, self.max_para_len],  # src_word_pos
                    [-1, self.max_para_num],  # src_sent_pos
                    [-1, self.max_para_num, self.max_para_len],  # src_words_slf_attn_bias
                    [-1, self.max_para_num],  # src_sents_slf_attn_bias
                    [-1, self.max_para_num, self.max_para_num],  # graph_attn_bias
                    [-1, 1],  # start_tokens
                    [-1, 1],  # init_scores
                    [-1],  # parent_idx
                    [-1, 1]],  # data_ids
            dtypes=['int64', 'int64', 'int64', 'float32', 'float32', 'float32',
                    'int64', 'float32', 'int64', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)

        (src_word, src_word_pos, src_sent_pos, src_words_slf_attn_bias, src_sents_slf_attn_bias,
         graph_attn_bias, start_tokens, init_scores, parent_idx, data_ids) = \
            fluid.layers.read_file(pyreader)

        src_words_slf_attn_bias = layers.expand(layers.unsqueeze(src_words_slf_attn_bias, axes=[2, 3]),
                                                expand_times=[1, 1, self._n_head, self.max_para_len, 1])
        src_words_slf_attn_bias.stop_gradient = True
        src_sents_slf_attn_bias = layers.expand(layers.unsqueeze(src_sents_slf_attn_bias, axes=[1, 2]),
                                                expand_times=[1, self._n_head, self.max_para_num, 1])
        src_sents_slf_attn_bias.stop_gradient = True
        graph_attn_bias = layers.expand(layers.unsqueeze(graph_attn_bias, axes=[1]),
                                        expand_times=[1, self._n_head, 1, 1])
        graph_attn_bias.stop_gradient = True

        tgt_src_words_attn_bias = layers.slice(src_words_slf_attn_bias, axes=[3], starts=[0], ends=[1])
        tgt_src_words_attn_bias.stop_gradient = True
        tgt_src_sents_attn_bias = layers.slice(src_sents_slf_attn_bias, axes=[2], starts=[0], ends=[1])
        tgt_src_sents_attn_bias.stop_gradient = True

        src_word = layers.reshape(src_word, [-1, self.max_para_num, self.max_para_len, 1])
        src_word_pos = layers.reshape(src_word_pos, [-1, self.max_para_num, self.max_para_len, 1])
        src_sent_pos = layers.reshape(src_sent_pos, [-1, self.max_para_num, 1])

        enc_input = (src_word, src_word_pos, src_sent_pos, src_words_slf_attn_bias,
                     src_sents_slf_attn_bias, graph_attn_bias)
        enc_words_output, enc_sents_output = self.encode(enc_input=enc_input)

        def beam_search():
            """Beam search function"""

            max_len = layers.fill_constant(shape=[1], dtype=start_tokens.dtype, value=self.max_out_len, force_cpu=True)
            min_len = layers.fill_constant(shape=[1], dtype=start_tokens.dtype, value=self.min_out_len)
            neg_inf = layers.fill_constant(shape=[1], dtype='float32', value=-INF)
            step_idx = layers.fill_constant(shape=[1], dtype=start_tokens.dtype, value=0, force_cpu=True)
            step_next_idx = layers.fill_constant(shape=[1], dtype=start_tokens.dtype, value=1, force_cpu=True)
            cond = layers.less_than(x=step_idx, y=max_len)  # default force_cpu=True
            while_op = layers.While(cond)
            # array states will be stored for each step.
            ids = layers.array_write(layers.reshape(start_tokens, (-1, 1)), step_idx)
            scores = layers.array_write(init_scores, step_idx)
            # cell states will be overwrited at each step.
            # caches contains states of history steps in decoder self-attention
            # and static encoder output projections in encoder-decoder attention
            # to reduce redundant computation.
            caches = [
                {
                    "k":  # for self attention
                        layers.fill_constant_batch_size_like(
                            input=start_tokens,
                            shape=[-1, self._n_head, 0, self._emb_size // self._n_head],
                            dtype=enc_words_output.dtype,
                            value=0),
                    "v":  # for self attention
                        layers.fill_constant_batch_size_like(
                            input=start_tokens,
                            shape=[-1, self._n_head, 0, self._emb_size // self._n_head],
                            dtype=enc_words_output.dtype,
                            value=0),
                    "static_k_word":  # for encoder-decoder attention
                        layers.create_tensor(dtype=enc_words_output.dtype),
                    "static_v_word":  # for encoder-decoder attention
                        layers.create_tensor(dtype=enc_words_output.dtype),
                    "static_k_sent":  # for encoder-decoder attention
                        layers.create_tensor(dtype=enc_sents_output.dtype),
                    "static_v_sent":  # for encoder-decoder attention
                        layers.create_tensor(dtype=enc_sents_output.dtype)
                } for i in range(self._dec_n_layer)
            ]

            trigram_blocking = TrigramBlocking(start_tokens, self.tokenizer,
                                               use_fp16=self._use_fp16, beam_size=self.beam_size)

            with while_op.block():
                pre_ids = layers.array_read(array=ids, i=step_idx)
                pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
                # Since beam_search_op dosen't enforce pre_ids' shape, we can do
                # inplace reshape here which actually change the shape of pre_ids.
                # pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
                pre_scores = layers.array_read(array=scores, i=step_idx)
                # gather cell states corresponding to selected parent
                pre_src_words_attn_bias = layers.gather(tgt_src_words_attn_bias, index=parent_idx)
                pre_src_sents_attn_bias = layers.gather(tgt_src_sents_attn_bias, index=parent_idx)
                pre_graph_attn_bias = layers.gather(graph_attn_bias, index=parent_idx)
                pre_pos = layers.elementwise_mul(
                    x=layers.fill_constant_batch_size_like(
                        input=pre_src_sents_attn_bias,  # cann't use lod tensor here
                        value=1,
                        shape=[-1, 1, 1],
                        dtype=pre_ids.dtype),
                    y=step_idx,
                    axis=0)

                logits = self.decode(dec_input=(pre_ids, pre_pos, None, pre_src_words_attn_bias,
                                                pre_src_sents_attn_bias, pre_graph_attn_bias),
                                     enc_words_output=enc_words_output,
                                     enc_sents_output=enc_sents_output,
                                     caches=caches,
                                     gather_idx=parent_idx)

                # prevent generating end token if length less than min_out_len
                eos_index = layers.fill_constant(shape=[layers.shape(logits)[0]],
                                                 dtype='int64',
                                                 value=self.eos_idx)
                eos_index = fluid.one_hot(eos_index, depth=self.voc_size)
                less_cond = layers.cast(layers.less_than(x=step_idx, y=min_len), dtype='float32')
                less_val = layers.elementwise_mul(less_cond, neg_inf)
                eos_val = layers.elementwise_mul(eos_index, less_val, axis=0)
                revised_logits = layers.elementwise_add(logits, eos_val, axis=0)

                # topK reduction across beams, also contain special handle of
                # end beams and end sentences(batch reduction)
                topk_scores, topk_indices = layers.topk(
                    input=layers.softmax(revised_logits), k=self.beam_size)

                # Roll-Back previous-scores for length-penalty
                # previous-scores has been length-penaltied, before this timestep length-penalty, need roll-back
                # because of doing this, we need store the length-penaltied score in `scores`
                # while calculating use the un-penaltied score
                # -> safe for step_idx == 0 (initialization state), because previous-score == 0
                pre_timestep_length_penalty = fluid.layers.pow(
                    ((5.0 + fluid.layers.cast(step_idx, pre_scores.dtype)) / 6.0), self.len_penalty)
                pre_scores_wo_len_penalty = fluid.layers.elementwise_mul(pre_scores, pre_timestep_length_penalty)

                # calc trigram-blocking delta scores for current alive sequence
                if self.block_trigram:
                    trigram_blocking.update_seq(pre_ids, parent_idx)
                    trigram_blocking.expand_cand_seq(topk_indices)
                    fluid.layers.py_func(func=trigram_blocking.blocking_forward,
                                         x=[trigram_blocking.cand_seq,
                                            trigram_blocking.id2is_full_token],
                                         out=trigram_blocking.delta_score_out,
                                         backward_func=None)
                    layers.Print(trigram_blocking.delta_score_out, summarize=100,
                                 message="trigram_blocking.delta_score_out")
                    pre_scores_wo_len_penalty = fluid.layers.elementwise_add(x=trigram_blocking.delta_score_out,
                                                                             y=pre_scores_wo_len_penalty,
                                                                             axis=0)
                # => [N, topk]

                accu_scores = layers.elementwise_add(
                    x=layers.log(topk_scores), y=pre_scores_wo_len_penalty, axis=0)

                cur_timestep_length_penalty = layers.pow(((5.0 + layers.cast(step_next_idx, accu_scores.dtype)) / 6.0),
                                                         self.len_penalty)
                curr_scores = layers.elementwise_div(accu_scores, cur_timestep_length_penalty)

                # beam_search op uses lod to differentiate branches.
                curr_scores = layers.lod_reset(curr_scores, pre_ids)
                topk_indices = layers.lod_reset(topk_indices, pre_ids)
                selected_ids, selected_scores, gather_idx = layers.beam_search(
                    pre_ids=pre_ids,
                    pre_scores=pre_scores,
                    ids=topk_indices,
                    scores=curr_scores,
                    beam_size=self.beam_size,
                    end_id=self.eos_idx,
                    return_parent_idx=True)

                layers.increment(x=step_idx, value=1.0, in_place=True)
                layers.increment(x=step_next_idx, value=1.0, in_place=True)
                # cell states(caches) have been updated in wrap_decoder,
                # only need to update beam search states here.
                layers.array_write(selected_ids, i=step_idx, array=ids)
                layers.array_write(selected_scores, i=step_idx, array=scores)
                layers.assign(gather_idx, parent_idx)
                layers.assign(pre_src_words_attn_bias, tgt_src_words_attn_bias)
                layers.assign(pre_src_sents_attn_bias, tgt_src_sents_attn_bias)
                layers.assign(pre_graph_attn_bias, graph_attn_bias)

                length_cond = layers.less_than(x=step_idx, y=max_len)
                finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
                layers.logical_and(x=length_cond, y=finish_cond, out=cond)

            finished_ids, finished_scores = layers.beam_search_decode(
                ids, scores, beam_size=self.beam_size, end_id=self.eos_idx)

            return finished_ids, finished_scores

        finished_ids, finished_scores = beam_search()

        graph_vars = {
            "finished_ids": finished_ids,
            "finished_scores": finished_scores,
            "data_ids": data_ids
        }

        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars
