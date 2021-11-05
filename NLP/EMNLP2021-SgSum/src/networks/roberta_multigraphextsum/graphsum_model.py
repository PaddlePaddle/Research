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
"""RoBertaGraphSum model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import itertools
from collections import namedtuple
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from models.encoder import transformer_encoder, graph_encoder
from models.encoder import self_attention_pooling_layer as att_pooling
from models.neural_modules import pre_process_layer
import numpy as np
from models.encoder import pretrained_graph_encoder
from roberta.roberta import RoBERTaModel

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

    def __init__(self, args, config, roberta_config, padding_idx, 
                 bos_idx, eos_idx, vocab_size):
        self.args = args
        self.roberta_config = roberta_config
        self._emb_size = roberta_config['hidden_size']
        self._n_head = roberta_config['num_attention_heads']
        self._max_position_seq_len = roberta_config['max_position_embeddings']

        self._enc_graph_layer = config['enc_graph_layers']
        self._dec_n_layer = config['dec_graph_layers']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._preprocess_command = config['preprocess_command']
        self._postprocess_command = config['postprocess_command']
        self._word_emb_name = config['word_embedding_name']
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
        self.voc_size = vocab_size
        self.max_para_len = args.max_para_len
        self.max_para_num = args.max_para_num
        self.max_doc_num = args.max_doc_num
        self.multidoc_total_para_num = self.max_doc_num * self.max_para_num
        self.graph_type = args.graph_type
        self.max_tgt_len = args.max_tgt_len
        self.len_penalty = args.len_penalty
        self.max_out_len = args.max_out_len
        self.min_out_len = args.min_out_len
        self.pos_win = args.pos_win
        self.sent_pos_num = args.max_para_num
        self.hyperparameter = 0.1
        self.c_sent_num = args.candidate_sentence_num
        self.s_sent_num = args.selected_sentence_num
        self.c_summary_num = len(list(itertools.combinations(list(range(self.c_sent_num)), self.s_sent_num)))

    def encode(self, enc_input, cls_ids):
        """Encoding the source input"""

        src_word, src_seg, src_word_pos, src_sen_pos, \
        src_words_slf_attn_bias, src_sents_slf_attn_bias, graph_attn_bias = enc_input
        
        roberta = RoBERTaModel(
                        src_ids=src_word,
                        position_ids=src_word_pos,
                        task_ids=src_seg,
                        input_mask=src_words_slf_attn_bias,
                        config=self.roberta_config,
                        pad_id=self._padding_idx,
                        use_fp16=self._use_fp16)

        self._word_emb_name = roberta.word_emb_name
        # (batch_size * doc_num, max_ntoken, emb_dim)
        enc_words_output = roberta.get_sequence_output()
        enc_words_output_reshaped = layers.reshape(x=enc_words_output, shape=[layers.shape(cls_ids)[0], layers.shape(cls_ids)[1], self.max_para_len, self._emb_size])
        # (batch_size, doc_num, max_nblock, emb_dim)
        sents_vec = layers.gather_nd(enc_words_output_reshaped, cls_ids)
        # obtain the embedding of sent pos
        src_sen_pos_enc = fluid.layers.embedding(src_sen_pos,  # (batch_size, doc_num, n_blocks, emb_dim)
                                                 size=[self.sent_pos_num, self._emb_size],
                                                 param_attr=fluid.ParamAttr(
                                                         name=self._enc_sen_pos_emb_name,
                                                         trainable=False))
        src_sen_pos_enc.stop_gradient = True
        sents_vec = layers.scale(x=sents_vec, scale=self._emb_size ** 0.5)
        sents_vec = sents_vec + src_sen_pos_enc

        # (batch_size, doc_num*max_nblock, emb_dim)
        reshape_sents_vec = layers.reshape(x=sents_vec, shape=[layers.shape(sents_vec)[0], -1, self._emb_size])
        
        # (batch_size, n_head, max_nblock*doc_num, max_nblock**doc_num)
        # graph_attn_bias = layers.reshape(x=graph_attn_bias, 
        #             shape=[layers.shape(graph_attn_bias)[0], layers.shape(graph_attn_bias)[1], 
        #             layers.shape(graph_attn_bias)[2]*layers.shape(graph_attn_bias)[4], 
        #             -1])
        # layers.Print(reshape_sents_vec, message="reshape_sents_vec", summarize=100)
        
        # the paragraph-level graph encoder
        # (batch_size, doc_num*n_block, emb_dim)
        flatten_enc_sents_out = pretrained_graph_encoder(
                        sents_vec=reshape_sents_vec,  # (batch_size, doc_num*max_nblock, emb_dim)
                        src_sents_slf_attn_bias=src_sents_slf_attn_bias,  # (batch_size, n_head, max_nblock*doc_num, max_nblock**doc_num)
                        graph_attn_bias=graph_attn_bias,  # (batch_size, n_head, max_nblock*doc_num, max_nblock**doc_num)
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
                        preprocess_cmd=self._preprocess_command,
                        postprocess_cmd=self._postprocess_command,
                        param_initializer=self._param_initializer,
                        name='roberta_graph_encoder')
        
        # (batch_size, doc_num, n_block, emb_dim)
        enc_sents_out = layers.reshape(x=flatten_enc_sents_out, 
                                    shape=[-1, layers.shape(sents_vec)[1], layers.shape(sents_vec)[2], self._emb_size])

        return enc_sents_out, flatten_enc_sents_out


    def build_model(self, enc_input, sent_labels, sent_labels_weight, cls_ids, cand_summary_combinations, labels_ids, labels_ids_weight,
                    summary_rank, summary_rank_high, summary_rank_low):
        """Build the model with source encoding and target decoding"""

        src_word, src_seg, src_word_pos, src_sen_pos, \
        src_words_slf_attn_bias, src_sents_slf_attn_bias, graph_attn_bias = enc_input

        enc_sen_output, flatten_enc_sent_output = self.encode(enc_input, cls_ids)
        # enc_sen_output = layers.reshape(x=enc_sen_output, shape=[layers.shape(cls_ids)[0] * layers.shape(cls_ids)[1] * layers.shape(cls_ids)[2], self._emb_size])

        # (batch_size, 1, emb_size)
        doc_embedding = att_pooling(enc_input=flatten_enc_sent_output, #(batch_size, doc_num*sent_num, emb_size)
                                             attn_bias=src_sents_slf_attn_bias, #(batch_size, n_head, doc_num*sent_num, doc_num*sent_num)
                                             n_head=self._n_head,
                                             d_key=self._emb_size // self._n_head,
                                             d_value=self._emb_size // self._n_head,
                                             d_model=self._emb_size,
                                             d_inner_hid=self._emb_size * 4,
                                             prepostprocess_dropout=self._prepostprocess_dropout,
                                             attention_dropout=self._attention_dropout,
                                             relu_dropout=self._prepostprocess_dropout,
                                             n_block=1,
                                             preprocess_cmd=self._preprocess_command,
                                             postprocess_cmd=self._postprocess_command,
                                             name='doc_att_pooling')

        batch_doc_embedding = layers.expand(layers.unsqueeze(doc_embedding, axes=[1]), 
                                        expand_times=[1, layers.shape(enc_sen_output)[1], layers.shape(enc_sen_output)[2], 1])
        sent_emb_with_doc_emb = layers.concat(input=[enc_sen_output, batch_doc_embedding], axis=3)

        sent_scores = layers.fc(input=sent_emb_with_doc_emb, size=1, num_flatten_dims=3, 
                                name="sent_score_fc",
                                param_attr=fluid.ParamAttr(
                                    name="sent_score_fc.w",
                                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                                bias_attr=fluid.ParamAttr(
                                    name='sent_score_fc.bias',
                                    initializer=fluid.initializer.Constant(value=0.0)))
        # layers.Print(sent_scores, message="sent_scores", summarize=100)
        # layers.Print(sent_labels, message="sent_labels", summarize=100)
        # layers.Print(sent_labels_weight, message="sent_labels_weight", summarize=100)

        correct_scores = layers.elementwise_mul(x=sent_scores, y=sent_labels_weight, axis=0)
        sigmoid_correct_scores = layers.sigmoid(correct_scores)
        sigmoid_correct_scores = layers.elementwise_mul(x=sigmoid_correct_scores, y=sent_labels_weight, axis=0)
        # layers.Print(correct_scores, message="correct_scores", summarize=100)

        reshaped_label_ids_weight = layers.reshape(labels_ids_weight, shape=[layers.shape(labels_ids_weight)[0], -1])
        reshaped_label_ids_weight.stop_gradient = True
        label_ids_attn_bias = (reshaped_label_ids_weight - 1) * (1e18)
        label_ids_attn_bias = layers.expand(layers.unsqueeze(label_ids_attn_bias, axes=[1, 2]), expand_times=[1, self._n_head, layers.shape(label_ids_attn_bias)[1], 1])
        label_ids_attn_bias.stop_gradient = True

        gold_summary_emb = layers.gather_nd(enc_sen_output, labels_ids) # (batch_size, doc_num, max_label_num, emb_size)
        gold_summary_emb = layers.reshape(gold_summary_emb, shape=[layers.shape(gold_summary_emb)[0], -1, self._emb_size]) # (batch_size, doc_num*max_nblock, emb_size)
        # (batch_size, doc_num*max_nblock, emb_size)
        gold_summary_emb = pretrained_graph_encoder(
                        sents_vec=gold_summary_emb,  # (batch_size, doc_num*max_nblock, emb_dim)
                        src_sents_slf_attn_bias=label_ids_attn_bias,  # (batch_size, n_head, max_nblock*doc_num, max_nblock**doc_num)
                        graph_attn_bias=graph_attn_bias,  # (batch_size, n_head, max_nblock*doc_num, max_nblock**doc_num)
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
                        preprocess_cmd=self._preprocess_command,
                        postprocess_cmd=self._postprocess_command,
                        param_initializer=self._param_initializer,
                        name='roberta_graph_encoder')
        expand_labels_ids_weight = layers.reshape(layers.expand(
                                            layers.reshape(labels_ids_weight, shape=[-1, layers.shape(labels_ids_weight)[1], layers.shape(labels_ids_weight)[2], 1]), # (batch_size, doc_num, max_nblock, 1)
                                            expand_times=[1, 1, 1, self._emb_size]),  # (batch_size, doc_num, max_nblock, emb_size)
                                            shape=[layers.shape(labels_ids_weight)[0], -1, self._emb_size]) # (batch_size, doc_num*max_nblock, emb_size)
        expand_labels_ids_weight.stop_gradient = True

        """
        summary-level loss
        """
        gold_summary_emb = layers.elementwise_mul(gold_summary_emb, expand_labels_ids_weight)
        gold_summary_emb = layers.reduce_sum(gold_summary_emb, dim=1)
        label_num = layers.reduce_sum(layers.reshape(labels_ids_weight, shape=[layers.shape(labels_ids_weight)[0], -1]), dim=1)
        gold_summary_emb = gold_summary_emb / label_num # (batch_size, emb_size)
        cos_with_gold_summary = layers.cos_sim(layers.reshape(doc_embedding, shape=[-1, self._emb_size]), gold_summary_emb)
        cos_with_gold_summary = layers.reduce_mean(cos_with_gold_summary, dim=0)

        """
        cross entropy loss
        """
        #cost = layers.square_error_cost(input=correct_scores, label=sent_labels)
        cost = layers.sigmoid_cross_entropy_with_logits(x=correct_scores, label=sent_labels)
        cost = layers.elementwise_mul(x=cost, y=sent_labels_weight, axis=0) # (batch_size, doc_num, sent_num, 1)
        sum_cost = layers.reduce_sum(cost, dim=2)
        sum_cost = layers.reduce_sum(sum_cost, dim=1) # (batch_size, 1)
        sent_num = layers.reduce_sum(sent_labels_weight, dim=2)
        sent_num = layers.reduce_sum(sent_num, dim=1)
        sent_num.stop_gradient = True
        batch_cost = layers.elementwise_div(sum_cost, sent_num)
        ce_loss = layers.mean(x=sum_cost)

        
        """
        summary-level loss
        """
        # (batch_size, doc_num, 9, 3, emb_size)
        summary_rank_high_emb = layers.gather_nd(enc_sen_output, summary_rank_high)
        # (batch_size, doc_num, 9, emb_size)
        summary_rank_high_emb = layers.reduce_mean(summary_rank_high_emb, dim=3)
        summary_rank_high_emb = layers.reshape(summary_rank_high_emb, shape=[-1, 9, self._emb_size])

        summary_rank_low_emb = layers.gather_nd(enc_sen_output, summary_rank_low)
        summary_rank_low_emb = layers.reduce_mean(summary_rank_low_emb, dim=3)
        summary_rank_low_emb = layers.reshape(summary_rank_low_emb, shape=[-1, 9, self._emb_size])
       
        summary_rank_doc_emb = layers.expand(doc_embedding, expand_times=[1, layers.shape(summary_rank_high_emb)[1], 1])
        summary_rank_doc_emb = layers.reshape(summary_rank_doc_emb, shape=[-1, self._emb_size])

        summary_rank_high_emb = layers.reshape(summary_rank_high_emb, shape=[-1, self._emb_size])
        summary_rank_low_emb = layers.reshape(summary_rank_low_emb, shape=[-1, self._emb_size])
        cos_with_summary_rank_high = layers.cos_sim(summary_rank_high_emb, summary_rank_doc_emb)
        cos_with_summary_rank_high = layers.reshape(cos_with_summary_rank_high, shape=[-1, 9, 1])
        cos_with_summary_rank_low = layers.cos_sim(summary_rank_low_emb, summary_rank_doc_emb)
        cos_with_summary_rank_low = layers.reshape(cos_with_summary_rank_low, shape=[-1, 9, 1])
        summary_rank_sub = layers.elementwise_sub(x=cos_with_summary_rank_low+0.01, y=cos_with_summary_rank_high)
        summary_rank_label = summary_rank_sub > 0
        summary_rank_loss = layers.elementwise_mul(x=layers.cast(x=summary_rank_label, dtype="float32"), y=summary_rank_sub)
        sum_summary_rank_loss = layers.reduce_sum(summary_rank_loss, dim=1)
        avg_summary_rank_loss = layers.mean(x=sum_summary_rank_loss)

        summary_loss = avg_summary_rank_loss
        

        avg_cost = ce_loss + (1 - cos_with_gold_summary) + 0.00001 * summary_loss
        
        graph_vars = {
            "loss": avg_cost,
            "sum_correct": batch_cost,
            "token_num": sent_num,
            "sentence_loss": ce_loss,
            "summary_loss": (1 - cos_with_gold_summary)
        }
        for k, v in graph_vars.items():
            v.persistable = True

        return graph_vars

    def create_model(self, pyreader_name, is_prediction=False):
        """Create the network"""

        if is_prediction:
            return self.ext_predict(pyreader_name)

        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, self.max_doc_num, self.max_para_len, 1],  # src_word
                    [-1, self.max_doc_num, self.max_para_len, 1],  # src_word_pos
                    [-1, self.max_doc_num, self.max_para_num, 1],  # src_sent_pos
                    [-1, self.max_doc_num, self.max_para_len],  # src_words_slf_attn_bias
                    [-1, self.multidoc_total_para_num, self.multidoc_total_para_num],  # src_sents_slf_attn_bias
                    [-1, self.max_doc_num, self.max_para_num, self.multidoc_total_para_num],  # graph_attn_bias
                    [-1, self.max_doc_num, self.max_para_num, 1],   # sent_labels
                    [-1, self.max_doc_num, self.max_para_num, 1],   # sent_labels_weight
                    [-1, self.max_doc_num, self.max_para_num, 3], # cls_ids
                    [-1, self.max_doc_num, self.max_para_num, 3], # sep_ids
                    [-1, self.c_summary_num, self.s_sent_num, 2], # cand_summary_combinations
                    [-1, self.max_doc_num, self.max_para_num, 3], # labels_ids
                    [-1, self.max_doc_num, self.max_para_num], # labels_ids_weight
                    [-1, self.max_doc_num, 10, 3, 3], # candidate_summary_rank
                    [-1, self.max_doc_num, 10, 3, 3], # summary_rank_high
                    [-1, self.max_doc_num, 10, 3, 3]], # summary_rank_low
            dtypes=['int64', 'int64', 'int64', 'float32', 'float32', 'float32',
                    'float32', 'float32', 'int64', 'int64', 'int64', 'int64', 'float32',
                    'int64', 'int64', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)

        (src_word, src_word_pos, src_sent_pos, 
         src_words_slf_attn_bias, src_sents_slf_attn_bias,
         graph_attn_bias, sent_labels, sent_labels_weight, cls_ids, src_seg,
         cand_summary_combinations,
         labels_ids, labels_ids_weight,
         summary_rank, summary_rank_high, summary_rank_low,) = fluid.layers.read_file(pyreader)
        # layers.Print(graph_attn_bias, message="graph_attn_bias", summarize=100)
        
        src_word = layers.reshape(x=src_word, shape=[-1, self.max_para_len, 1])
        src_word_pos = layers.reshape(x=src_word_pos, shape=[-1, self.max_para_len, 1])
        src_words_slf_attn_bias = layers.expand(layers.unsqueeze(src_words_slf_attn_bias, axes=[2, 3]), 
                                                 expand_times=[1, 1, self._n_head, self.max_para_len, 1])
        src_words_slf_attn_bias = layers.reshape(x=src_words_slf_attn_bias, shape=[-1, self._n_head, self.max_para_len, self.max_para_len])
        src_words_slf_attn_bias.stop_gradient = True

        # (batch_size, n_head, doc_num*max_nblock, doc_num*max_nblock)
        src_sents_slf_attn_bias = layers.expand(layers.unsqueeze(src_sents_slf_attn_bias, axes=[1]),
                                                expand_times=[1, self._n_head, 1, 1])
        src_sents_slf_attn_bias.stop_gradient = True

        # (batch_size, n_head, max_nblock*doc_num, max_nblock*doc_num)
        graph_attn_bias = layers.expand(layers.unsqueeze(graph_attn_bias, axes=[1]),
                                        expand_times=[1, self._n_head, 1, 1, 1])
        graph_attn_bias = layers.reshape(graph_attn_bias, shape=[layers.shape(graph_attn_bias)[0], self._n_head, -1, layers.shape(graph_attn_bias)[4]])
        graph_attn_bias.stop_gradient = True

        enc_input = (src_word, src_seg, src_word_pos, src_sent_pos, 
                     src_words_slf_attn_bias, src_sents_slf_attn_bias, graph_attn_bias)
        # layers.Print(graph_attn_bias, message="graph_attn_bias", summarize=100)

        graph_vars = self.build_model(enc_input=enc_input, sent_labels=sent_labels, 
                                      sent_labels_weight=sent_labels_weight, cls_ids=cls_ids, 
                                      cand_summary_combinations=cand_summary_combinations,
                                      labels_ids=labels_ids, labels_ids_weight=labels_ids_weight,
                                      summary_rank=summary_rank, summary_rank_high=summary_rank_high, summary_rank_low=summary_rank_low)
        return pyreader, graph_vars
    
    def ext_predict(self, pyreader_name, is_prediction=False):
        """Create predict network"""
        
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, self.max_doc_num, self.max_para_len, 1],  # src_word
                    [-1, self.max_doc_num, self.max_para_len, 1],  # src_word_pos
                    [-1, self.max_doc_num, self.max_para_num, 1],  # src_sent_pos
                    [-1, self.max_doc_num, self.max_para_len],  # src_words_slf_attn_bias
                    [-1, self.multidoc_total_para_num, self.multidoc_total_para_num],  # src_sents_slf_attn_bias
                    [-1, self.max_doc_num, self.max_para_num, self.multidoc_total_para_num],  # graph_attn_bias
                    [-1, self.max_doc_num, self.max_para_num, 1],   # sent_labels
                    [-1, self.max_doc_num, self.max_para_num, 1],   # sent_labels_weight
                    [-1, 1],                   # data_ids
                    [-1, self.max_doc_num, self.max_para_num, 3],# cls_ids
                    [-1, self.max_doc_num, self.max_para_num, 3],  # sep_ids
                    [-1, self.c_summary_num, self.s_sent_num, 2]], # cand_summary_combinations
            dtypes=['int64', 'int64', 'int64', 'float32', 'float32', 'float32',
                    'float32', 'float32', 'int64', 'int64', 'int64', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            name=pyreader_name,
            use_double_buffer=True)

        (src_word, src_word_pos, src_sent_pos,
         src_words_slf_attn_bias, src_sents_slf_attn_bias,
         graph_attn_bias, sent_labels, sent_labels_weight, data_ids, cls_ids, src_seg,
         cand_summary_combinations) = \
            fluid.layers.read_file(pyreader)
        
        src_word = layers.reshape(x=src_word, shape=[-1, self.max_para_len, 1])
        src_word_pos = layers.reshape(x=src_word_pos, shape=[-1, self.max_para_len, 1])
        src_words_slf_attn_bias = layers.expand(layers.unsqueeze(src_words_slf_attn_bias, axes=[2, 3]), 
                                                 expand_times=[1, 1, self._n_head, self.max_para_len, 1])
        src_words_slf_attn_bias = layers.reshape(x=src_words_slf_attn_bias, shape=[-1, self._n_head, self.max_para_len, self.max_para_len])
        src_words_slf_attn_bias.stop_gradient = True

        # (batch_size, n_head, max_nblock*doc_num, max_nblock*doc_num)
        src_sents_slf_attn_bias = layers.expand(layers.unsqueeze(src_sents_slf_attn_bias, axes=[1]),
                                                expand_times=[1, self._n_head, 1, 1])
        src_sents_slf_attn_bias.stop_gradient = True
        # layers.Print(src_sents_slf_attn_bias, message="after", summarize=100)
        
        # (batch_size, n_head, max_nblock*doc_num, max_nblock*doc_num)
        graph_attn_bias = layers.expand(layers.unsqueeze(graph_attn_bias, axes=[1]),
                                        expand_times=[1, self._n_head, 1, 1, 1])
        graph_attn_bias = layers.reshape(graph_attn_bias, shape=[layers.shape(graph_attn_bias)[0], self._n_head, -1, layers.shape(graph_attn_bias)[4]])
        graph_attn_bias.stop_gradient = True

        enc_input = (src_word, src_seg, src_word_pos, src_sent_pos, 
                     src_words_slf_attn_bias, src_sents_slf_attn_bias, graph_attn_bias)
        
        enc_sen_output, flatten_enc_sent_output = self.encode(enc_input, cls_ids)

        # (batch_size, 1, emb_size)
        doc_embedding = att_pooling(enc_input=flatten_enc_sent_output, #(batch_size, doc_num*sent_num, emb_size)
                                             attn_bias=src_sents_slf_attn_bias, #(batch_size, n_head, doc_num*sent_num, doc_num*sent_num)
                                             n_head=self._n_head,
                                             d_key=self._emb_size // self._n_head,
                                             d_value=self._emb_size // self._n_head,
                                             d_model=self._emb_size,
                                             d_inner_hid=self._emb_size * 4,
                                             prepostprocess_dropout=self._prepostprocess_dropout,
                                             attention_dropout=self._attention_dropout,
                                             relu_dropout=self._prepostprocess_dropout,
                                             n_block=1,
                                             preprocess_cmd=self._preprocess_command,
                                             postprocess_cmd=self._postprocess_command,
                                             name='doc_att_pooling')
           
        batch_doc_embedding = layers.expand(layers.unsqueeze(doc_embedding, axes=[1]), 
                                        expand_times=[1, layers.shape(enc_sen_output)[1], layers.shape(enc_sen_output)[2], 1])
        sent_emb_with_doc_emb = layers.concat(input=[enc_sen_output, batch_doc_embedding], axis=3)

        sent_scores = layers.fc(input=sent_emb_with_doc_emb, size=1, num_flatten_dims=3, 
                                name="sent_score_fc", act="sigmoid",
                                param_attr=fluid.ParamAttr(
                                    name="sent_score_fc.w",
                                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                                bias_attr=fluid.ParamAttr(
                                    name='sent_score_fc.bias',
                                    initializer=fluid.initializer.Constant(value=0.0)))
        # (batch_size, doc_num, sent_num, 1)
        correct_scores = layers.elementwise_mul(x=sent_scores, y=sent_labels_weight, axis=0)
        
        reshaped_correct_scores = layers.reshape(correct_scores, shape=[layers.shape(correct_scores)[0], -1, 1])
        
        score_rank = layers.argsort(input=reshaped_correct_scores, axis=1, descending=True)[1]
        cand_summary_sent_ids = score_rank[:,:self.c_sent_num,]
        sent_id_combins = layers.gather_nd(cand_summary_sent_ids, cand_summary_combinations)
        selected_sent_attn_bias = fluid.one_hot(input=layers.reshape(sent_id_combins, shape=[layers.shape(sent_id_combins)[0], self.c_summary_num, -1]), 
                                        depth=layers.shape(flatten_enc_sent_output)[1]) # (batch_size, c_summary_num, selected_sent_num, doc_num*max_nblock)
        selected_sent_attn_bias = layers.reduce_sum(selected_sent_attn_bias, dim=2) # (batch_size, c_summary_num, doc_num*max_nblock)
        # layers.Print(selected_sent_attn_bias, message="selected_sent_attn_bias", summarize=100)
        selected_sent_attn_bias = layers.expand(layers.unsqueeze(selected_sent_attn_bias, axes=[2, 3]), 
                        expand_times=[1, 1, self._n_head, layers.shape(selected_sent_attn_bias)[-1], 1]) # (batch_size, c_summary_num, n_head, doc_num*max_nblock, doc_num*max_nblock)
        selected_sent_attn_bias = layers.reshape(selected_sent_attn_bias, 
                shape=[-1, self._n_head, layers.shape(selected_sent_attn_bias)[-1], layers.shape(selected_sent_attn_bias)[-1]]) # (batch_size*c_summary_num, n_head, doc_num*max_nblock, doc_num*max_nblock)
        selected_sent_attn_bias.stop_gradient = True

        # layers.Print(selected_sent_attn_bias, message="selected_sent_attn_bias", summarize=100)


        # pad_sent_attn_bias = fluid.layers.zeros(shape=[layers.shape(selected_sent_attn_bias)[0], layers.shape(selected_sent_attn_bias)[1], 
        #                 (layers.shape(selected_sent_attn_bias)[3] - layers.shape(selected_sent_attn_bias)[2]), layers.shape(selected_sent_attn_bias)[3]], dtype='float32')
        # # layers.Print(selected_sent_attn_bias, message="selected_sent_attn_bias", summarize=100)
        # # layers.Print(pad_sent_attn_bias, message="pad_sent_attn_bias", summarize=-1)
        # sent_attn_bias = layers.concat(input=[selected_sent_attn_bias, pad_sent_attn_bias], axis=2) # (batch_size, c_summary_num, doc_num*max_nblock, doc_num*max_nblock)
        # # layers.Print(selected_sent_attn_bias, message="selected_sent_attn_bias", summarize=100)
        # sent_attn_bias = layers.expand(layers.unsqueeze(layers.reshape(sent_attn_bias, shape=[-1, layers.shape(sent_attn_bias)[2], layers.shape(sent_attn_bias)[3]]),
        #                                 axes=[1]), expand_times=[1, self._n_head, 1, 1]) # (batch_size*c_summary_num, n_head, doc_num*max_nblock, doc_num*max_nblock)
        # # layers.Print(sent_attn_bias, message="sent_attn_bias", summarize=100)


        graph_attn_bias = layers.expand(graph_attn_bias, expand_times=[self.c_summary_num, 1, 1, 1]) # (batch_size*c_summary_num, n_head, doc_num*max_nblock, doc_num*max_nblock)
        graph_attn_bias.stop_gradient = True
        # layers.Print(graph_attn_bias, message="graph_attn_bias", summarize=100)
        # layers.Print(sent_id_combins, message="before", summarize=100)
        pad_sent_combins_matrix = layers.zeros(shape=[layers.shape(sent_id_combins)[0], self.c_summary_num, layers.shape(graph_attn_bias)[2] - self.s_sent_num, 1], dtype='int64')
        # layers.Print(pad_sent_combins_matrix, message="pad_sent_combins_matrix", summarize=100)
        pad_sent_id_combins = layers.concat(input=[sent_id_combins, pad_sent_combins_matrix], axis=2) # (batch_size, c_summary_num, doc_num*max_nblock, 1)
        # layers.Print(sent_id_combins, message="after", summarize=100)


        batch_id = layers.range(0, layers.shape(correct_scores)[0], 1, 'int64')
        batch_id = layers.reshape(batch_id, shape=[-1, 1])
        batch_id = layers.expand(batch_id, expand_times=[1, self.c_summary_num*layers.shape(pad_sent_id_combins)[2]])
        batch_id = layers.reshape(batch_id, shape=[-1, self.c_summary_num, layers.shape(pad_sent_id_combins)[2], 1]) # (batch_size, c_summary_num, doc_num*max_nblock, 1)
        
        combined_sent_id = layers.concat(input=[batch_id, pad_sent_id_combins], axis=-1)
        # layers.Print(combined_sent_id, message="combined_sent_id", summarize=-1)

        # c_summary_emb = layers.gather_nd(flatten_enc_sent_output, combined_sent_id) # (batch_size, c_summary_num, doc_num*max_nblock, emb_size)
        # layers.Print(flatten_enc_sent_output, message="flatten_enc_sent_output", summarize=-1)
        c_summary_emb = layers.expand(layers.unsqueeze(flatten_enc_sent_output, axes=[1]), expand_times=[1, self.c_summary_num, 1, 1]) # (batch_size, c_summary_num, doc_num*max_nblock, emb_size)
        c_summary_emb = layers.reshape(c_summary_emb, shape=[-1, layers.shape(c_summary_emb)[2], self._emb_size]) # (batch_size*c_summary_num, doc_num*max_nblock, emb_dim)
        # layers.Print(c_summary_emb, message="test", summarize=-1)
        # (batch_size*c_summary_num, doc_num*max_nblock, emb_dim)
        c_summary_emb = pretrained_graph_encoder(
                        sents_vec=c_summary_emb,  # (batch_size*c_summary_num, doc_num*max_nblock, emb_dim)
                        src_sents_slf_attn_bias=(selected_sent_attn_bias - 1) * 1e18,  # (batch_size*c_summary_num, n_head, max_nblock*doc_num, max_nblock**doc_num)
                        graph_attn_bias=graph_attn_bias,  # (batch_size*c_summary_num, n_head, max_nblock*doc_num, max_nblock**doc_num)
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
                        preprocess_cmd=self._preprocess_command,
                        postprocess_cmd=self._postprocess_command,
                        param_initializer=self._param_initializer,
                        name='roberta_graph_encoder')
        c_summary_emb = layers.reshape(c_summary_emb, shape=[-1, self.c_summary_num, layers.shape(c_summary_emb)[1], self._emb_size]) # (batch_size, c_summary_num, doc_num*max_nblock, emb_dim)
        # layers.Print(c_summary_emb, message="after", summarize=100)
        # sent_attn_bias = layers.reshape(layers.squeeze(sent_attn_bias, axes=[1, 3]), shape=[-1, self.c_summary_num, layers.shape(c_summary_emb)[2], 1])
        # layers.Print(selected_sent_attn_bias, message="before", summarize=100)
        sent_attn_bias = selected_sent_attn_bias[:,:1,:1,:]
        sent_attn_bias = layers.reshape(layers.squeeze(sent_attn_bias, axes=[1, 2]), shape=[-1, self.c_summary_num, layers.shape(c_summary_emb)[2], 1]) # (batch_size, c_summary_num, doc_num*max_nblock, 1)
        sent_attn_bias.stop_gradient = True
        # layers.Print(sent_attn_bias, message="sent_attn_bias", summarize=-1)
        expand_sent_attn_bias = layers.expand(sent_attn_bias, expand_times=[1, 1, 1, self._emb_size]) # (batch_size, c_summary_num, doc_num*max_nblock, emb_size)
        expand_sent_attn_bias.stop_gradient = True
        c_summary_emb = layers.elementwise_mul(c_summary_emb, expand_sent_attn_bias)
        # layers.Print(c_summary_emb, message="c_summary_emb", summarize=-1)
        c_summary_emb = layers.reduce_sum(c_summary_emb, dim=2) # (batch_size, c_summary_num, emb_size)
        c_summary_emb = c_summary_emb / self.s_sent_num

        c_summary_doc_emb = layers.expand(doc_embedding, expand_times=[1, self.c_summary_num, 1])
        c_summary_doc_emb = layers.reshape(c_summary_doc_emb, shape=[-1, self._emb_size]) # (batch_size*c_summary_num, emb_size)
        reshape_c_summary_emb = layers.reshape(c_summary_emb, shape=[-1, self._emb_size])
        cos_with_c_summary = layers.cos_sim(reshape_c_summary_emb, c_summary_doc_emb)
        cos_with_c_summary = layers.reshape(cos_with_c_summary, shape=[-1, self.c_summary_num])
        # layers.Print(cos_with_c_summary, message="cos_with_c_summary", summarize=100)
        # layers.Print(sent_id_combins, message="sent_id_combines", summarize=100)


        c_summary_rank = layers.argsort(input=cos_with_c_summary, axis=1, descending=True)[1]
        c_summary = c_summary_rank[:,:1]
        
        # batch_id = layers.range(0, layers.shape(c_summary)[0], 1, 'int64')
        # batch_id = layers.reshape(batch_id, shape=[-1, 1])
        # batch_id = layers.expand(batch_id, expand_times=[1, layers.shape(c_summary)[1]])
        # combined_summary_id = layers.concat(input=[batch_id, c_summary], axis=-1)
        # best_summary_emb = layers.gather_nd(c_summary_emb, combined_summary_id) # (batch_size, emb_size)

        graph_vars = {
            "sent_scores": correct_scores,
            "data_ids": data_ids,
            "cand_summary_sent_ids": sent_id_combins,
            "best_summary_index": c_summary
        }
        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars