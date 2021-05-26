#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""GRAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import paddle.fluid as fluid
from model.graph_encoder import encoder, pre_process_layer

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())


class GRANModel(object):
    """
    GRAN model class.
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 edge_labels,
                 config,
                 weight_sharing=True,
                 use_fp16=False):

        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._emb_size = config['hidden_size']
        self._intermediate_size = config['intermediate_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_dropout_prob']

        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']
        self._n_edge = config['num_edges']
        self._max_seq_len = config['max_seq_len']
        self._max_arity = config['max_arity']
        self._e_soft_label = config['entity_soft_label']
        self._r_soft_label = config['relation_soft_label']
        self._weight_sharing = weight_sharing

        self._node_emb_name = "node_embedding"
        self._edge_emb_name_k = "edge_embedding_key"
        self._edge_emb_name_v = "edge_embedding_value"
        self._dtype = "float16" if use_fp16 else "float32"

        # Initialize all weights by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(input_ids, input_mask, edge_labels)

    def _build_model(self, input_ids, input_mask, edge_labels):
        # get node embeddings of input tokens
        emb_out = fluid.layers.embedding(
            input=input_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._node_emb_name, initializer=self._param_initializer),
            is_sparse=False)
        emb_out = pre_process_layer(
            emb_out, 'nd', self._prepostprocess_dropout, name='pre_encoder')

        # get edge embeddings between input tokens
        edge_labels = fluid.layers.reshape(
            x=edge_labels, shape=[-1, 1], inplace=True)
        edges_key = fluid.layers.embedding(
            input=edge_labels,
            size=[self._n_edge, self._emb_size // self._n_head],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._edge_emb_name_k,
                initializer=self._param_initializer),
            is_sparse=False)
        edges_value = fluid.layers.embedding(
            input=edge_labels,
            size=[self._n_edge, self._emb_size // self._n_head],
            dtype=self._dtype,
            param_attr=fluid.ParamAttr(
                name=self._edge_emb_name_v,
                initializer=self._param_initializer),
            is_sparse=False)
        edge_mask = fluid.layers.sign(
            fluid.layers.cast(
                x=edge_labels, dtype='float32'))
        # edge_mask.stop_gradient = True
        edges_key = fluid.layers.elementwise_mul(
            x=edges_key, y=edge_mask, axis=0)
        edges_key = fluid.layers.reshape(
            x=edges_key,
            shape=[self._max_seq_len, self._max_seq_len, -1],
            inplace=True)
        edges_value = fluid.layers.elementwise_mul(
            x=edges_value, y=edge_mask, axis=0)
        edges_value = fluid.layers.reshape(
            x=edges_value,
            shape=[self._max_seq_len, self._max_seq_len, -1],
            inplace=True)

        # get multi-head self-attention mask
        if self._dtype == "float16":
            input_mask = fluid.layers.cast(x=input_mask, dtype=self._dtype)

        self_attn_mask = fluid.layers.matmul(
            x=input_mask, y=input_mask, transpose_y=True)
        self_attn_mask = fluid.layers.scale(
            x=self_attn_mask,
            scale=1000000.0,
            bias=-1.0,
            bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        # stack of graph transformer encoders
        self._enc_out = encoder(
            enc_input=emb_out,
            edges_key=edges_key,
            edges_value=edges_value,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            name='encoder')

    def get_sequence_output(self):
        return self._enc_out

    def get_mask_lm_output(self, mask_pos, mask_label, mask_type):
        """
        Get the loss & logits for masked entity/relation prediction.
        """
        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        reshaped_emb_out = fluid.layers.reshape(
            x=self._enc_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_fc.w_0',
                initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        # transform: layer norm
        mask_trans_feat = pre_process_layer(
            mask_trans_feat, 'n', name='mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self._node_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self._voc_size],
                dtype=self._dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)
        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)

        special_indicator = fluid.layers.fill_constant_batch_size_like(
            input=mask_label, shape=[-1, 2], dtype='int64', value=-1)
        relation_indicator = fluid.layers.fill_constant_batch_size_like(
            input=mask_label,
            shape=[-1, self._n_relation],
            dtype='int64',
            value=-1)
        entity_indicator = fluid.layers.fill_constant_batch_size_like(
            input=mask_label,
            shape=[-1, (self._voc_size - self._n_relation - 2)],
            dtype='int64',
            value=1)
        type_indicator = fluid.layers.concat(
            input=[relation_indicator, entity_indicator], axis=-1)
        type_indicator = fluid.layers.elementwise_mul(
            x=type_indicator, y=mask_type, axis=0)
        type_indicator = fluid.layers.concat(
            input=[special_indicator, type_indicator], axis=-1)
        type_indicator = fluid.layers.cast(x=type_indicator, dtype='float32')
        type_indicator = fluid.layers.thresholded_relu(
            x=type_indicator, threshold=0.0)

        fc_out_mask = fluid.layers.scale(
            x=type_indicator,
            scale=1000000.0,
            bias=-1.0,
            bias_after_scale=False)
        fc_out = fluid.layers.elementwise_add(x=fc_out, y=fc_out_mask)

        one_hot_labels = fluid.layers.one_hot(
            input=mask_label, depth=self._voc_size)
        type_indicator = fluid.layers.elementwise_sub(
            x=type_indicator, y=one_hot_labels)
        num_candidates = fluid.layers.reduce_sum(input=type_indicator, dim=-1)

        mask_type = fluid.layers.cast(x=mask_type, dtype='float32')
        soft_labels = ((1 + mask_type) * self._e_soft_label +
                       (1 - mask_type) * self._r_soft_label) / 2.0
        soft_labels = fluid.layers.expand(soft_labels, [1, self._voc_size])
        soft_labels = soft_labels * one_hot_labels + (1.0 - soft_labels) * \
                      fluid.layers.elementwise_div(x=type_indicator, y=num_candidates, axis=0)
        soft_labels.stop_gradient = True

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(
            logits=fc_out, label=soft_labels, soft_label=True)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        return mean_mask_lm_loss, fc_out
