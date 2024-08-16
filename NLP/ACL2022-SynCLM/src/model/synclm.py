#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""SynCLM model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import paddle.fluid as fluid
from model.transformer_encoder import encoder, pre_process_layer, pos_encoder


class SynCLMConfig(object):

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing SynCLM model config file '%s'" % config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key, None)

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class SynCLMModel(object):

    def __init__(self, emb_ids, input_mask, config, weight_sharing=True, att_layers=[-1]):

        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing

        self._emb_vocab_size = {"word_embedding": self._voc_size, "pos_embedding": self._max_position_seq_len}

        self._sent_types = config['type_vocab_size']
        self._emb_vocab_size["sent_embedding"] = self._sent_types

        self._emb_dtype = "float32"
        self.caches = None

        self.att_layers = att_layers
        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(scale=config['initializer_range'])

        self._build_model(emb_ids, input_mask)

    def _gen_input(self, emb_ids, input_mask):
        emb_out = None
        for emb_name, emb_id in emb_ids.items():
            if emb_name == "sent_embedding":
                continue  # don't use sentence embedding in roberta
            emb = fluid.layers.embedding(input=emb_id,
                                         size=[self._emb_vocab_size[emb_name], self._emb_size],
                                         dtype=self._emb_dtype,
                                         param_attr=fluid.ParamAttr(name=emb_name, initializer=self._param_initializer))
            emb_out = emb_out + emb if emb_out else emb

        emb_out = pre_process_layer(emb_out, 'nd', self._prepostprocess_dropout, name="pre_encoder")

        self_attn_mask = input_mask
        self_attn_mask = fluid.layers.scale(x=self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        return emb_out, n_head_self_attn_mask

    def encode(self, emb_ids, input_mask):
        # padding id in vocabulary must be set to 0

        emb_out, n_head_self_attn_mask = self._gen_input(emb_ids, input_mask)
        enc_out, att_mats = encoder(enc_input=emb_out,
                                    attn_bias=n_head_self_attn_mask,
                                    n_layer=self._n_layer,
                                    n_head=self._n_head,
                                    d_key=self._emb_size // self._n_head,
                                    d_value=self._emb_size // self._n_head,
                                    d_model=self._emb_size,
                                    d_inner_hid=self._emb_size * 4,
                                    prepostprocess_dropout=self._prepostprocess_dropout,
                                    attention_dropout=self._attention_dropout,
                                    relu_dropout=0,
                                    hidden_act=self._hidden_act,
                                    preprocess_cmd="",
                                    postprocess_cmd="dan",
                                    param_initializer=self._param_initializer,
                                    name='encoder',
                                    caches=self.caches)

        return enc_out, att_mats

    def _build_model(self, emb_ids, input_mask):
        self._enc_out, self.att_mats = self.encode(emb_ids, input_mask)

    def get_sequence_output(self):
        return self._enc_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(input=self._enc_out, axes=[1], starts=[0], ends=[1])

        next_sent_feat = fluid.layers.fc(input=next_sent_feat,
                                         size=self._emb_size,
                                         param_attr=fluid.ParamAttr(name="mask_lm_trans_fc.w_0",
                                                                    initializer=self._param_initializer),
                                         bias_attr="mask_lm_trans_fc.b_0")
        return next_sent_feat

    def get_mask_loss(self, mask_label, mask_pos):
        """Get the loss & accuracy for pretraining"""
        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')
        reshaped_emb_out = fluid.layers.reshape(x=self._enc_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(input=mask_feat,
                                          size=self._emb_size,
                                          act=self._hidden_act,
                                          param_attr=fluid.ParamAttr(name='mask_lm_trans_fc.w_0',
                                                                     initializer=self._param_initializer),
                                          bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))
        # transform: layer norm
        mask_trans_feat = fluid.layers.layer_norm(mask_trans_feat,
                                                  begin_norm_axis=len(mask_trans_feat.shape) - 1,
                                                  param_attr=fluid.ParamAttr(
                                                      name='mask_lm_trans_layer_norm_scale',
                                                      initializer=fluid.initializer.Constant(1.)),
                                                  bias_attr=fluid.ParamAttr(name='mask_lm_trans_layer_norm_bias',
                                                                            initializer=fluid.initializer.Constant(1.)))

        mask_lm_out_bias_attr = fluid.ParamAttr(name="mask_lm_out_fc.b_0",
                                                initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = fluid.layers.matmul(x=mask_trans_feat,
                                         y=fluid.default_main_program().global_block().var("word_embedding"),
                                         transpose_y=True)
            fc_out += fluid.layers.create_parameter(shape=[self._voc_size],
                                                    dtype=self._emb_dtype,
                                                    attr=mask_lm_out_bias_attr,
                                                    is_bias=True)
        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(name="mask_lm_out_fc.w_0",
                                                                initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(logits=fc_out, label=mask_label)
        mask_lm_loss = fluid.layers.reduce_sum(mask_lm_loss, dim=1)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)
        return reshaped_emb_out, mean_mask_lm_loss

    def get_phrase_loss(self, samples, positives, negatives, negatives_mask, input_mask, max_neg_num):
        samples = fluid.layers.cast(x=samples, dtype='int32')
        positives = fluid.layers.cast(x=positives, dtype='int32')
        reshaped_positives = fluid.layers.reshape(positives, shape=[-1, 1])

        negatives = fluid.layers.cast(x=negatives, dtype='int32')
        reshaped_negatives = fluid.layers.reshape(negatives, shape=[-1, 1])

        att_mat = None
        att_mask = input_mask
        for i in self.att_layers:
            if att_mat is None:
                att_mat = self.att_mats[i]
            else:
                att_mat += self.att_mats[i]
        att_mat = att_mat / len(self.att_layers)
        att_mat = fluid.layers.reduce_mean(att_mat, 1)
        reshaped_att_out = fluid.layers.reshape(x=att_mat, shape=[-1, 0])
        reshaped_att_mask = fluid.layers.reshape(x=att_mask, shape=[-1, 0])
        sam_feat = fluid.layers.gather(input=reshaped_att_out, index=samples)
        att_mask = fluid.layers.gather(input=reshaped_att_mask, index=samples)

        pos_feat = fluid.layers.gather(input=reshaped_att_out, index=reshaped_positives)

        neg_feat = fluid.layers.gather(input=reshaped_att_out, index=reshaped_negatives)
        neg_feat = fluid.layers.unsqueeze(neg_feat, axes=[1])
        neg_feat = fluid.layers.reshape(neg_feat, shape=[-1, max_neg_num, 0])

        # x_h, x_in, x_out have the same att_mask.
        pos_jsd = self.jsd(sam_feat, pos_feat, att_mask)
        neg_jsd = self.jsd_negs(sam_feat, neg_feat, att_mask)

        phrase_tau = fluid.layers.create_parameter(name="phrase_tau",
                                                   shape=[1],
                                                   dtype='float32',
                                                   default_initializer=fluid.initializer.Constant(0.1))
        phrase_tau.stop_gradient = True
        pos_jsd = fluid.layers.exp(pos_jsd / phrase_tau)
        neg_jsd = fluid.layers.exp(neg_jsd / phrase_tau)
        neg_jsd = fluid.layers.reduce_sum(negatives_mask * neg_jsd, dim=1)
        phrase_loss = fluid.layers.log(pos_jsd / (pos_jsd + neg_jsd))
        mean_phrase_loss = fluid.layers.mean(phrase_loss)
        return mean_phrase_loss, phrase_tau

    def get_tree_loss(self, reshaped_emb_out, samples, positives, positives_mask, negatives, negatives_mask, input_mask,
                      max_neg_num, max_sub_num):
        """Get the loss & accuracy for pretraining"""
        samples = fluid.layers.cast(x=samples, dtype='int32')
        positives = fluid.layers.cast(x=positives, dtype='int32')
        reshaped_positives = fluid.layers.reshape(positives, shape=[-1, 1])
        negatives = fluid.layers.cast(x=negatives, dtype='int32')
        reshaped_negatives = fluid.layers.reshape(negatives, shape=[-1, 1])

        # batch, hidden
        samples_feat = fluid.layers.gather(input=reshaped_emb_out, index=samples)
        # batch * max_sub_num, hidden
        positives_feat = fluid.layers.gather(input=reshaped_emb_out, index=reshaped_positives)
        # batch * max_neg_num * max_sub_num, hidden
        negatives_feat = fluid.layers.gather(input=reshaped_emb_out, index=reshaped_negatives)

        # transform: fc
        contrast_trans_fc_w0 = fluid.ParamAttr(name='contrast_trans_fc.w_0', initializer=self._param_initializer)
        contrast_trans_fc_b0 = fluid.ParamAttr(name='contrast_trans_fc.b_0')
        contrast_trans_layer_norm_scale = fluid.ParamAttr(name='contrast_trans_layer_norm_scale',
                                                          initializer=fluid.initializer.Constant(1.))
        contrast_trans_layer_norm_bias = fluid.ParamAttr(name='contrast_trans_layer_norm_bias',
                                                         initializer=fluid.initializer.Constant(1.))

        samples_feat = fluid.layers.fc(input=samples_feat,
                                       size=self._emb_size,
                                       act=self._hidden_act,
                                       param_attr=contrast_trans_fc_w0,
                                       bias_attr=contrast_trans_fc_b0)
        positives_feat = fluid.layers.fc(input=positives_feat,
                                         size=self._emb_size,
                                         act=self._hidden_act,
                                         param_attr=contrast_trans_fc_w0,
                                         bias_attr=contrast_trans_fc_b0)
        negatives_feat = fluid.layers.fc(input=negatives_feat,
                                         size=self._emb_size,
                                         act=self._hidden_act,
                                         param_attr=contrast_trans_fc_w0,
                                         bias_attr=contrast_trans_fc_b0)

        # transform: layer norm
        samples_feat = fluid.layers.layer_norm(samples_feat,
                                               begin_norm_axis=len(samples_feat.shape) - 1,
                                               param_attr=contrast_trans_layer_norm_scale,
                                               bias_attr=contrast_trans_layer_norm_bias)
        positives_feat = fluid.layers.layer_norm(positives_feat,
                                                 begin_norm_axis=len(positives_feat.shape) - 1,
                                                 param_attr=contrast_trans_layer_norm_scale,
                                                 bias_attr=contrast_trans_layer_norm_bias)
        negatives_feat = fluid.layers.layer_norm(negatives_feat,
                                                 begin_norm_axis=len(negatives_feat.shape) - 1,
                                                 param_attr=contrast_trans_layer_norm_scale,
                                                 bias_attr=contrast_trans_layer_norm_bias)

        contrast_out_w = fluid.ParamAttr(name="contrast_out_fc.w_0", initializer=self._param_initializer)
        contrast_out_bias_attr = fluid.ParamAttr(name="contrast_out_fc.b_0",
                                                 initializer=fluid.initializer.Constant(value=0.0))
        samples_feat = fluid.layers.fc(input=samples_feat,
                                       size=self._emb_size,
                                       param_attr=contrast_out_w,
                                       bias_attr=contrast_out_bias_attr)
        positives_feat = fluid.layers.fc(input=positives_feat,
                                         size=self._emb_size,
                                         param_attr=contrast_out_w,
                                         bias_attr=contrast_out_bias_attr)
        negatives_feat = fluid.layers.fc(input=negatives_feat,
                                         size=self._emb_size,
                                         param_attr=contrast_out_w,
                                         bias_attr=contrast_out_bias_attr)

        # batch, max_sub_num, hidden
        positives_feat = fluid.layers.reshape(x=positives_feat, shape=[*positives.shape[:-1], self._emb_size])
        # batch, mat_neg_num * max_sub_num, 768
        negatives_feat = fluid.layers.reshape(x=negatives_feat,
                                              shape=[-1, negatives.shape[1] * negatives.shape[2], self._emb_size])
        # batch, 1, hidden
        samples_feat = fluid.layers.unsqueeze(samples_feat, axes=[1])
        # batch, 1, max_sub_num
        positive_att = fluid.layers.matmul(samples_feat, positives_feat, transpose_y=True)
        positive_attn_bias = fluid.layers.scale(x=positives_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        positive_attn_bias = fluid.layers.unsqueeze(positive_attn_bias, axes=[1])
        # batch, 1, max_sub_num
        positive_att += positive_attn_bias
        positive_weights = fluid.layers.softmax(positive_att)
        positives_feat = fluid.layers.matmul(positive_weights, positives_feat)
        # batch, 1, max_neg_num * (max_sub_num + 1)
        negatives_att = fluid.layers.matmul(samples_feat, negatives_feat, transpose_y=True)
        # batch, max_neg_num, (max_sub_num + 1)
        negatives_att = fluid.layers.reshape(x=negatives_att, shape=[0, max_neg_num, max_sub_num + 1])
        negative_attn_bias = fluid.layers.scale(x=negatives_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        negatives_att += negative_attn_bias
        # batch, max_neg_num, (max_sub_num + 1)
        negative_weights = fluid.layers.softmax(negatives_att)
        # batch, max_neg_num, max_sub_num + 1, 768
        negatives_feat = fluid.layers.reshape(x=negatives_feat, shape=[0, max_neg_num, max_sub_num + 1, self._emb_size])
        # batch, max_neg_num, 1, (max_sub_num + 1)
        negative_weights = fluid.layers.unsqueeze(input=negative_weights, axes=[2])
        # batch, max_neg_num, 1, 768
        negatives_feat = fluid.layers.matmul(negative_weights, negatives_feat)

        def cos_sim(x, y):
            product_x_y = fluid.layers.reduce_sum(x * y, dim=-1)
            product_x_square = fluid.layers.reduce_sum(fluid.layers.square(x), dim=-1)
            product_y_square = fluid.layers.reduce_sum(fluid.layers.square(y), dim=-1)
            cosine = product_x_y / fluid.layers.sqrt(product_x_square * product_y_square)
            return cosine

        tau_con = fluid.layers.create_parameter(name="tau_contrast",
                                                shape=[1],
                                                dtype='float32',
                                                default_initializer=fluid.initializer.Constant(0.1))
        tau_con.stop_gradient = True
        pos_cos = cos_sim(samples_feat, positives_feat)
        pos_cos = fluid.layers.exp(pos_cos / tau_con)

        samples_feat_for_neg = fluid.layers.stack([samples_feat] * max_neg_num, axis=1)
        neg_cos = cos_sim(samples_feat_for_neg, negatives_feat)
        neg_cos = fluid.layers.exp(neg_cos / tau_con)
        negatives_mask = fluid.layers.reduce_sum(negatives_mask, dim=2) > 0
        neg_cos = fluid.layers.squeeze(neg_cos, axes=[2])
        negatives_mask = fluid.layers.cast(negatives_mask, dtype='float32')
        neg_cos, _ = fluid.layers.topk(negatives_mask * neg_cos, 3)
        neg_cos = fluid.layers.reduce_sum(neg_cos, dim=1, keep_dim=True)
        contrast_loss = -1 * fluid.layers.log(pos_cos / (pos_cos + neg_cos))
        mean_contrast_loss = fluid.layers.mean(contrast_loss)
        return mean_contrast_loss, tau_con

    def kl(self, x, y, mask, eps=1e-6):
        product = x * fluid.layers.log((x + eps) / (y + eps))
        product = product * mask
        kl = fluid.layers.reduce_sum(product, -1)
        return kl

    def jsd(self, x, y, mask):
        m = (x + y) / 2
        jsd = (self.kl(x, m, mask) + self.kl(y, m, mask)) / 2
        return jsd

    def jsd_negs(self, x, y, mask):
        x = fluid.layers.stack([x] * y.shape[1], axis=1)
        mask = fluid.layers.stack([mask] * y.shape[1], axis=1)
        m = (x + y) / 2
        jsd = (self.kl(x, m, mask) + self.kl(y, m, mask)) / 2
        return jsd

    def index_sample(self, x, index):
        """Select input value according to index
        
        Arags：
            input: input matrix
            index: index matrix
        Returns:
            output
        >>> input
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
        >>> index
        [
            [1, 2],
            [0, 1]
        ]
        >>> index_sample(input, index)
        [
            [2, 3],
            [4, 5]
        ]
        """
        x_s = x.shape
        r_x = x
        index = fluid.layers.unsqueeze(index, axes=-1)
        # generate arange index, shape like index
        # arr_index = layers.arange(start=0, end=layers.cast(layers.shape(x)[0], ), dtype=index.dtype)
        zero = fluid.layers.fill_constant(shape=[1], dtype=index.dtype, value=0)
        one = fluid.layers.fill_constant(shape=[1], dtype=index.dtype, value=1)
        batch_size = fluid.layers.cast(fluid.layers.reduce_sum(fluid.layers.ones_like(index)),
                                       dtype=index.dtype) / index.shape[1]
        arr_index = fluid.layers.unsqueeze(fluid.layers.range(zero, batch_size, one, dtype=index.dtype), [1, 2])

        arr_index = fluid.layers.expand_as(arr_index, index)
        #  genrate new index
        new_index = fluid.layers.concat([arr_index, index], -1)
        new_index = fluid.layers.reshape(new_index, (-1, 2))
        # get output
        out = fluid.layers.gather_nd(r_x, new_index)
        out = fluid.layers.reshape(out, (-1, index.shape[1]))
        return out
