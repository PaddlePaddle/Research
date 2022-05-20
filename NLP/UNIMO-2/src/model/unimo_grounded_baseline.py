#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

"""
File: unimo_grounded_baseline.py
Author: liwei(liwei85@baidu.com)
Date: 2021-08-31 20:46
Desc: RoBERTa + ViT + Grounded Transformer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import paddle
import paddle.distributed.fleet as fleet
from model.transformer_encoder import encoder as grounded_encoder
from model.transformer_encoder import encoder as text_encoder
from model.transformer_encoder import pre_process_layer as text_pre_process_layer
from model.vision_transformer_encoder import encoder as vit_encoder
from model.vision_transformer_encoder import pre_process_layer as vit_pre_process_layer
from utils.pos_emb_interpolate import interpolate_pos_embed


class VlConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing VL model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key, None)

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        for arg, value in self._config_dict.items():
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class VlModel(object):
    def __init__(self,
                 image_input=None,
                 image_mask=None,
                 text_input=None,
                 text_mask=None,
                 config=None,
                 weight_sharing=True,
                 task_type="normal",
                 decoding=False,
                 gather_idx=None,
                 grounded_encoder_trainable=True,
                 vit_encoder_trainable=True,
                 text_encoder_trainable=True,
                 with_cmcl_projection=True,
                 text_enc_layers=None,
                 grounding_enc_layers=None):

        # for grounded cross-modal encoder
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._max_position_seq_len = config['max_position_embeddings']
        self._hidden_act = config['hidden_act']
        self._emb_size = config['hidden_size']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._emb_dtype = "float32"
        self._sent_types = config['type_vocab_size']

        # for text encoder
        self._text_n_layer = config['text_num_hidden_layers']
        self._text_n_head = config['text_num_attention_heads']
        self._text_voc_size = config['text_vocab_size']
        self._text_max_position_seq_len = config['text_max_position_embeddings']
        self._text_hidden_act = config['text_hidden_act']
        self._text_prepostprocess_dropout = config['text_hidden_dropout_prob']
        self._text_attention_dropout = config['text_attention_probs_dropout_prob']
        self._text_emb_vocab_size = {"text.word_embedding": self._text_voc_size,
                                     "text.pos_embedding": self._text_max_position_seq_len}

        # for vit encoder
        self._vit_n_layer = config['vit_num_hidden_layers']
        self._vit_n_head = config['vit_num_attention_heads']
        self._vit_hidden_act = config['vit_hidden_act']
        self._vit_prepostprocess_dropout = config['vit_hidden_dropout_prob']
        self._vit_attention_dropout = config['vit_attention_probs_dropout_prob']
        self._vit_layer_norm_eps = config['vit_layer_norm_eps']

        self._weight_sharing = weight_sharing
        self._grounded_encoder_trainable = grounded_encoder_trainable
        self._vit_encoder_trainable = vit_encoder_trainable
        self._text_encoder_trainable = text_encoder_trainable
        self._with_cmcl_projection = with_cmcl_projection

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = paddle.fluid.initializer.TruncatedNormalInitializer(scale=config['initializer_range'])
        self._bias_initializer = paddle.fluid.initializer.ConstantInitializer(value=0.0)

        assert text_input is not None or image_input is not None, "text_input and image_input cannot be both None"
        self._task_type = task_type
        self._is_img2txt_task = (task_type == "img2txt")
        self._is_multimodal_task = (image_input is not None)

        self.image_size = config['image_size']
        self.num_codebook = config['num_codebook']
        self.resolution = config['resolution']
        self.width = self.image_size // config['resolution']
        self.patch_seq_len = self.image_size * self.image_size // (config['resolution'] * config['resolution'])
        self.patch_emb_size = config['resolution'] * config['resolution'] * 3

        if text_enc_layers is None:
            text_enc_layers = list(range(self._text_n_layer))
        if grounding_enc_layers is None:
            grounding_enc_layers = list(range(self._n_layer))

        print("text_enc_layers", text_enc_layers)
        print("grounding_enc_layers", grounding_enc_layers)
        self.text_enc_layers = text_enc_layers
        self.grounding_enc_layers = grounding_enc_layers

        self.cmcl_temperature = paddle.static.create_parameter(
            shape=[1],
            dtype=self._emb_dtype,
            attr=paddle.ParamAttr(
                name="cmcl_temperature",
                initializer=paddle.fluid.initializer.ConstantInitializer(value=0.07)))

        if decoding:
            self.grounded_caches = [{
                "k":
                    paddle.fluid.layers.fill_constant_batch_size_like(
                        input=text_input["text.word_embedding"] if text_input is not None else image_input[
                            "pixel_embedding"],
                        shape=[-1, 0, self._emb_size],
                        dtype=self._emb_dtype,  # float32,
                        value=0),
                "v":
                    paddle.fluid.layers.fill_constant_batch_size_like(
                        input=text_input["text.word_embedding"] if text_input is not None else image_input[
                            "pixel_embedding"],
                        shape=[-1, 0, self._emb_size],
                        dtype=self._emb_dtype,  # float32,
                        value=0),
            } for i in range(self._n_layer)]

            self.text_caches = [{
                "k":
                    paddle.fluid.layers.fill_constant_batch_size_like(
                        input=text_input["text.word_embedding"] if text_input is not None else image_input[
                            "pixel_embedding"],
                        shape=[-1, 0, self._emb_size],
                        dtype=self._emb_dtype,  # float32,
                        value=0),
                "v":
                    paddle.fluid.layers.fill_constant_batch_size_like(
                        input=text_input["text.word_embedding"] if text_input is not None else image_input[
                            "pixel_embedding"],
                        shape=[-1, 0, self._emb_size],
                        dtype=self._emb_dtype,  # float32,
                        value=0),
            } for i in range(self._text_n_layer)]
        else:
            self.grounded_caches = None
            self.text_caches = None

        self._build_model(text_input=text_input,
                          text_mask=text_mask,
                          gather_idx=gather_idx,
                          image_input=image_input,
                          image_mask=image_mask)

    def _build_model(self, text_input=None, text_mask=None, gather_idx=None, image_input=None, image_mask=None):
        if text_input is None and image_input is not None:  # for img2txt when decoding or image tasks
            self._enc_v_out, self.all_checkpoints = self.encode(image_input=image_input,
                                                                image_mask=image_mask,
                                                                gather_idx=gather_idx)
        elif text_input is not None and image_input is None:  # for textual tasks
            self._enc_l_out, self.all_checkpoints = self.encode(text_input=text_input,
                                                                text_mask=text_mask,
                                                                gather_idx=gather_idx)
        else:  # for multi-modal tasks
            self._enc_v_out, self._enc_l_out, self.all_checkpoints = \
                self.encode(text_input=text_input,
                            text_mask=text_mask,
                            gather_idx=gather_idx,
                            image_input=image_input,
                            image_mask=image_mask)

    def encode(self, text_input=None, text_mask=None, gather_idx=None,
               image_input=None, image_mask=None, decoding_step=False, grounded_decoding_mask=None):
        all_checkpoints = []
        # padding id in vocabulary must be set to 0
        if text_input is None and image_input is not None:  # for img2txt task when decoding or image tasks
            emb_v_out, v_seq_len, n_head_self_attn_mask, _checkpoints = \
                self._gen_input(image_input=image_input, image_mask=image_mask, gather_idx=gather_idx)
            all_checkpoints.extend(_checkpoints)

            enc_v_out, grounding_checkpoints = grounded_encoder(
                enc_input=emb_v_out,
                attn_bias=n_head_self_attn_mask,
                enc_layers=self.grounding_enc_layers,
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
                name='grounded.encoder',
                caches=self.grounded_caches,
                gather_idx=gather_idx,
                trainable=self._grounded_encoder_trainable)
            all_checkpoints.extend(grounding_checkpoints)
            return enc_v_out, all_checkpoints

        elif image_input is None and text_input is not None:  # for textual task
            if decoding_step:  # for step-by-step generation during decoding
                emb_l_out, l_seq_len, n_head_self_attn_mask, _checkpoints = \
                    self._gen_input(text_input=text_input, text_mask=text_mask, gather_idx=gather_idx,
                                    decoding_step=True, grounded_decoding_mask=grounded_decoding_mask)
                all_checkpoints.extend(_checkpoints)

                enc_l_out, grounding_checkpoints = grounded_encoder(
                    enc_input=emb_l_out,
                    attn_bias=n_head_self_attn_mask,
                    enc_layers=self.grounding_enc_layers,
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
                    name='grounded.encoder',
                    caches=self.grounded_caches,
                    gather_idx=gather_idx,
                    trainable=self._grounded_encoder_trainable)
                all_checkpoints.extend(grounding_checkpoints)
                return enc_l_out, all_checkpoints

            else:
                emb_l_out, l_seq_len, n_head_self_attn_mask, _checkpoints = \
                    self._gen_input(text_input=text_input, text_mask=text_mask, gather_idx=gather_idx)
                all_checkpoints.extend(_checkpoints)

                enc_l_out, grounding_checkpoints = grounded_encoder(
                    enc_input=emb_l_out,
                    attn_bias=n_head_self_attn_mask,
                    enc_layers=self.grounding_enc_layers,
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
                    name='grounded.encoder',
                    caches=self.grounded_caches,
                    gather_idx=gather_idx,
                    trainable=self._grounded_encoder_trainable)
                all_checkpoints.extend(grounding_checkpoints)
                return enc_l_out, all_checkpoints

        elif image_input is not None and text_input is not None:  # for multi-modal task
            emb_v_out, emb_l_out, v_seq_len, l_seq_len, n_head_self_attn_mask, _checkpoints = \
                self._gen_input(image_input=image_input, image_mask=image_mask,
                                text_input=text_input, text_mask=text_mask, gather_idx=gather_idx)
            all_checkpoints.extend(_checkpoints)
            emb_vl_out = paddle.concat([emb_v_out, emb_l_out], axis=1)

            enc_vl_out, grounding_checkpoints = grounded_encoder(
                enc_input=emb_vl_out,
                attn_bias=n_head_self_attn_mask,
                enc_layers=self.grounding_enc_layers,
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
                name='grounded.encoder',
                caches=self.grounded_caches,
                gather_idx=gather_idx,
                trainable=self._grounded_encoder_trainable)
            all_checkpoints.extend(grounding_checkpoints)
            enc_v_out = paddle.slice(
                input=enc_vl_out, axes=[1], starts=[0], ends=[v_seq_len])
            enc_l_out = paddle.slice(
                input=enc_vl_out, axes=[1], starts=[v_seq_len], ends=[v_seq_len + l_seq_len])
            return enc_v_out, enc_l_out, all_checkpoints
        else:
            raise ValueError("The input is invalid")

    def vit_encode(self, image_input, image_mask):
        """encode image by pre-trained ViT"""
        assert image_mask is not None, "text_mask should not be none"
        image_self_attn_mask = paddle.matmul(x=paddle.transpose(image_mask, perm=[0, 2, 1]), y=image_mask)
        self_attn_mask = paddle.scale(
            x=image_self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.stack(
            x=[self_attn_mask] * self._vit_n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        pixel_embeddings = paddle.static.nn.conv2d(
            input=image_input['pixel_embedding'],
            num_filters=self._emb_size,
            filter_size=self.resolution,
            stride=self.resolution,
            padding=(self.resolution - 1) // 2,
            param_attr=paddle.ParamAttr(
                name="vit.patch_embeddings_projection_weight",
                trainable=self._vit_encoder_trainable),
            bias_attr=paddle.ParamAttr(
                name="vit.patch_embeddings_projection_bias",
                trainable=self._vit_encoder_trainable),
            data_format="NHWC")
        # paddle.static.Print(paddle.shape(pixel_embeddings), message="pixel_embeddings", summarize=-1)
        pixel_embeddings = paddle.reshape(pixel_embeddings, shape=[-1, self.patch_seq_len, self._emb_size])

        cls_token_emb = paddle.static.create_parameter(
            shape=[1, 1, self._emb_size],
            dtype=self._emb_dtype,
            attr=paddle.ParamAttr(
                name="vit.cls_token_embeddings",
                trainable=self._vit_encoder_trainable,
                initializer=self._param_initializer))
        cls_token_emb = paddle.expand(x=cls_token_emb,
                                      shape=[paddle.shape(pixel_embeddings)[0], 1, self._emb_size])

        # cpncate global [CLS] token with image patches
        # (batch_size, patch_seq_len + 1, emb_dim)
        all_pixel_embeddings = paddle.concat(x=[cls_token_emb, pixel_embeddings], axis=1)

        # default image_size=224, resolution=16, patch_seq_len=196
        pixel_pos_emb = paddle.static.create_parameter(
            shape=[1, 197, self._emb_size],
            dtype=self._emb_dtype,
            attr=paddle.ParamAttr(name="vit.position_embeddings",
                                  trainable=self._vit_encoder_trainable,
                                  initializer=self._param_initializer))
        # paddle.static.Print(paddle.shape(pixel_pos_emb), message="pixel_pos_emb", summarize=-1)
        if self.patch_seq_len > 196:  # when image_size > 224
            pixel_pos_emb = interpolate_pos_embed(pixel_pos_emb, self.patch_seq_len)

        emb_v_out = all_pixel_embeddings + pixel_pos_emb
        emb_v_out = vit_pre_process_layer(
            emb_v_out, 'd', self._vit_prepostprocess_dropout, name='vit.pre_encoder',
            trainable=self._vit_encoder_trainable)

        vit_enc_out, checkpoints = vit_encoder(
            enc_input=emb_v_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._vit_n_layer,
            n_head=self._vit_n_head,
            d_key=self._emb_size // self._vit_n_head,
            d_value=self._emb_size // self._vit_n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._vit_prepostprocess_dropout,
            attention_dropout=self._vit_attention_dropout,
            relu_dropout=0,
            hidden_act=self._vit_hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=self._param_initializer,
            name='vit.encoder',
            trainable=self._vit_encoder_trainable)
        vit_seq_len = paddle.shape(vit_enc_out)[1]  # patch_seq_len + 1

        return vit_enc_out, vit_seq_len, n_head_self_attn_mask, checkpoints

    def text_encode(self, text_input, text_mask, gather_idx=None, decoding_step=False):
        assert text_mask is not None, "text_mask should not be none"
        if decoding_step:
            text_self_attn_mask = text_mask
        else:
            text_self_attn_mask = paddle.matmul(x=paddle.transpose(text_mask, perm=[0, 2, 1]), y=text_mask)

        self_attn_mask = paddle.scale(
            x=text_self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.stack(
            x=[self_attn_mask] * self._n_head, axis=1)
        n_head_self_attn_mask.stop_gradient = True

        # text part
        text_emb = paddle.static.nn.embedding(
            input=text_input["text.word_embedding"],
            size=[self._text_emb_vocab_size["text.word_embedding"], self._emb_size],
            dtype=self._emb_dtype,
            param_attr=paddle.ParamAttr(
                name='text.word_embedding',
                trainable=self._text_encoder_trainable,
                initializer=self._param_initializer))
        text_emb = paddle.squeeze(text_emb, axis=2)  # (batch_size, seq_len, emb_dim)

        pos_emb = paddle.static.nn.embedding(
            input=text_input["text.pos_embedding"],
            size=[self._text_emb_vocab_size["text.pos_embedding"], self._emb_size],
            dtype=self._emb_dtype,
            param_attr=paddle.ParamAttr(
                name='text.pos_embedding',
                trainable=self._text_encoder_trainable,
                initializer=self._param_initializer))
        pos_emb = paddle.squeeze(pos_emb, axis=2)  # (batch_size, seq_len, emb_dim)

        emb_out = text_emb + pos_emb
        emb_out = text_pre_process_layer(
            emb_out, 'nd', self._text_prepostprocess_dropout,
            name="text.pre_encoder", trainable=self._text_encoder_trainable)

        text_enc_out, checkpoints = text_encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            enc_layers=self.text_enc_layers,
            n_head=self._text_n_head,
            d_key=self._emb_size // self._text_n_head,
            d_value=self._emb_size // self._text_n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._text_prepostprocess_dropout,
            attention_dropout=self._text_attention_dropout,
            relu_dropout=0,
            hidden_act=self._text_hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            name='text.encoder',
            caches=self.text_caches,
            gather_idx=gather_idx,
            trainable=self._text_encoder_trainable)

        text_seq_len = paddle.shape(text_enc_out)[1]
        return text_enc_out, text_seq_len, n_head_self_attn_mask, checkpoints

    def _gen_input(self, text_input=None, text_mask=None,
                   image_input=None, image_mask=None, gather_idx=None, decoding_step=False,
                   grounded_decoding_mask=None):
        _checkpoints = []
        """encode images and texts independently by Vit and RoBERTa, get the optimal grounded tokens"""
        if image_input is not None:
            # visual part
            self.vit_enc_out, vit_seq_len, image_self_attn_mask, vit_checkpoints = self.vit_encode(image_input,
                                                                                                   image_mask)
            _checkpoints.extend(vit_checkpoints)

        if text_input is not None:
            # textual part
            self.text_enc_out, text_seq_len, text_self_attn_mask, text_checkpoints = self.text_encode(text_input,
                                                                                                      text_mask,
                                                                                                      gather_idx=gather_idx,
                                                                                                      decoding_step=decoding_step)
            _checkpoints.extend(text_checkpoints)

        if image_input is not None and text_input is not None:
            vl_mask = paddle.concat([image_mask, text_mask], axis=2)
            vl_self_attn_mask = paddle.matmul(paddle.transpose(vl_mask, perm=[0, 2, 1]), vl_mask)
            vl_self_attn_mask = paddle.scale(
                x=vl_self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
            n_head_self_attn_mask = paddle.stack(
                x=[vl_self_attn_mask] * self._n_head, axis=1)
            n_head_self_attn_mask.stop_gradient = True

            return self.vit_enc_out, self.text_enc_out, vit_seq_len, text_seq_len, n_head_self_attn_mask, _checkpoints

        elif image_input is not None and text_input is None:
            return self.vit_enc_out, vit_seq_len, image_self_attn_mask, _checkpoints

        elif image_input is None and text_input is not None:
            if decoding_step:
                assert grounded_decoding_mask is not None, "grounded_decoding_mask should not be none"
                self_attn_mask = paddle.scale(
                    x=grounded_decoding_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
                n_head_self_attn_mask = paddle.stack(
                    x=[self_attn_mask] * self._n_head, axis=1)
                n_head_self_attn_mask.stop_gradient = True

                return self.text_enc_out, text_seq_len, n_head_self_attn_mask, _checkpoints
            else:
                return self.text_enc_out, text_seq_len, text_self_attn_mask, _checkpoints

    def get_checkpoints(self):
        return self.all_checkpoints

    def get_text_encoder_output(self):
        return self.text_enc_out

    def get_vit_encoder_output(self):
        return self.vit_enc_out

    def get_text_sequence_output(self):
        return self._enc_l_out

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = paddle.slice(
            input=self._enc_l_out, axes=[1], starts=[0], ends=[1])
        next_sent_feat = paddle.reshape(
            x=next_sent_feat, shape=[-1, self._emb_size])
        next_sent_feat = paddle.static.nn.fc(
            x=next_sent_feat,
            size=self._emb_size,
            activation="relu",
            weight_attr=paddle.ParamAttr(
                name="grounded.pooled_fc_text.w_0", initializer=self._param_initializer),
            bias_attr="grounded.pooled_fc_text.b_0")

        next_sent_v_feat = paddle.slice(
            input=self._enc_v_out, axes=[1], starts=[0], ends=[1])
        next_sent_v_feat = paddle.reshape(
            x=next_sent_v_feat, shape=[-1, self._emb_size])
        next_sent_v_feat = paddle.static.nn.fc(
            x=next_sent_v_feat,
            size=self._emb_size,
            activation="relu",
            weight_attr=paddle.ParamAttr(
                name="grounded.pooled_fc_image.w_0", initializer=self._param_initializer),
            bias_attr="grounded.pooled_fc_image.b_0")
        return next_sent_feat, next_sent_v_feat

    def get_pooled_txt_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = paddle.slice(
            input=self._enc_l_out, axes=[1], starts=[0], ends=[1])
        next_sent_feat = paddle.reshape(
            x=next_sent_feat, shape=[-1, self._emb_size])
        next_sent_feat = paddle.static.nn.fc(
            x=next_sent_feat,
            size=self._emb_size,
            activation="relu",
            weight_attr=paddle.ParamAttr(
                name="grounded.pooled_fc_text.w_0", initializer=self._param_initializer),
            bias_attr="grounded.pooled_fc_text.b_0")

        return next_sent_feat

    def get_pooled_img_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_v_feat = paddle.slice(
            input=self._enc_v_out, axes=[1], starts=[0], ends=[1])
        next_sent_v_feat = paddle.reshape(
            x=next_sent_v_feat, shape=[-1, self._emb_size])
        next_sent_v_feat = paddle.static.nn.fc(
            x=next_sent_v_feat,
            size=self._emb_size,
            activation="relu",
            weight_attr=paddle.ParamAttr(
                name="grounded.pooled_fc_image.w_0", initializer=self._param_initializer),
            bias_attr="grounded.pooled_fc_image.b_0")
        return next_sent_v_feat

    def get_match_output(self, text, image, mode="mul"):
        if mode == "sum":
            emb_fuse = text + image
        elif mode == "mul":
            emb_fuse = text * image
        else:
            print("current mode %s is not supported" % mode)
            return
        emb_fuse = paddle.fluid.layers.nn.dropout(emb_fuse,
                                                  self._attention_dropout,
                                                  dropout_implementation="upscale_in_train")
        return emb_fuse

    def get_img_txt_matching_output(self, labels, is_multimodal=None, num_labels=2, fusion_method="mul"):
        print('-----img_txt_matching------')
        self.next_sent_feat, self.next_sent_v_feat = self.get_pooled_output()
        next_feat = self.get_match_output(self.next_sent_feat,
                                          self.next_sent_v_feat, fusion_method)
        # paddle.static.Print(next_feat, message='next_feat', summarize=-1)

        matching_fc_out = paddle.static.nn.fc(
            x=next_feat,
            size=num_labels,
            weight_attr=paddle.ParamAttr(
                name="grounded.img_text_matching_fc.w_0",
                initializer=self._param_initializer),
            bias_attr="grounded.img_text_matching_fc.b_0")
        # paddle.static.Print(matching_fc_out, message='matching_fc_out', summarize=-1)

        matching_loss, matching_softmax = paddle.nn.functional.softmax_with_cross_entropy(
            logits=matching_fc_out, label=labels, return_softmax=True)
        # paddle.static.Print(matching_loss, message='matching_loss', summarize=-1)
        # paddle.static.Print(matching_softmax, message='matching_softmax', summarize=-1)

        image_match_weight = paddle.cast(is_multimodal, dtype="float32")
        matching_loss = paddle.multiply(x=matching_loss, y=image_match_weight)  # exclude only text input
        # paddle.static.Print(matching_loss, message='matching_loss_2', summarize=-1)
        avg_img_num = paddle.mean(x=image_match_weight)
        # paddle.static.Print(avg_img_num, message='avg_img_num', summarize=-1)
        mean_matching_loss = paddle.mean(x=matching_loss) / (avg_img_num + 1e-18)
        # paddle.static.Print(mean_matching_loss, message='mean_matching_loss', summarize=-1)

        valid_condition = paddle.cast(paddle.reshape(x=is_multimodal, shape=[-1]), 'bool')
        valid_img2txt_ind = paddle.fluid.layers.where(valid_condition)
        res = paddle.is_empty(x=valid_img2txt_ind)

        def false_func():
            valid_matching_softmax = paddle.gather(x=matching_softmax, index=valid_img2txt_ind)
            valid_labels = paddle.gather(x=labels, index=valid_img2txt_ind)
            valid_labels = paddle.reshape(x=valid_labels, shape=[-1, 1])
            matching_acc = paddle.metric.accuracy(input=valid_matching_softmax, label=valid_labels)
            return matching_acc

        def true_func():
            return paddle.zeros(shape=[1], dtype='float32')

        matching_acc = paddle.static.nn.cond(res, true_fn=true_func, false_fn=false_func)
        return mean_matching_loss, matching_acc

    def get_pretraining_txt_output(self, mask_label, mask_pos, lm_loss_flag,
                                   with_text=None, is_multimodal=None, mm_lr_weight=1.0):
        """Get the loss & accuracy for pretraining language"""
        mask_pos = paddle.cast(x=mask_pos, dtype='int32')
        reshaped_emb_out = paddle.reshape(x=self._enc_l_out, shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = paddle.gather(x=reshaped_emb_out, index=mask_pos)

        with_text = paddle.fluid.layers.expand(with_text,  # [batch_size, max_seq_len]
                                               expand_times=[1, paddle.shape(self._enc_l_out)[1]])
        with_text = paddle.reshape(x=with_text, shape=[-1])  # [batch_size * max_seq_len]
        with_text = paddle.gather(x=with_text, index=mask_pos)
        with_text = paddle.reshape(x=with_text, shape=[-1, 1])  # [batch_size * max_seq_len, 1]
        with_text.stop_gradient = True

        is_multimodal = paddle.fluid.layers.expand(is_multimodal,  # [batch_size, max_seq_len]
                                                   expand_times=[1, paddle.shape(self._enc_l_out)[1]])
        is_multimodal = paddle.reshape(x=is_multimodal, shape=[-1])  # [batch_size * max_seq_len]
        is_multimodal = paddle.gather(x=is_multimodal, index=mask_pos)
        mm_loss_weight = is_multimodal * mm_lr_weight + (1 - is_multimodal)
        mm_loss_weight.stop_gradient = True

        # transform: fc
        mask_trans_feat = paddle.static.nn.fc(
            x=mask_feat,
            size=self._emb_size,
            activation=self._hidden_act,
            weight_attr=paddle.ParamAttr(
                name='grounded.mask_lm_trans_fc.w_0',
                trainable=self._grounded_encoder_trainable,
                initializer=self._param_initializer),
            bias_attr=paddle.ParamAttr(
                name='grounded.mask_lm_trans_fc.b_0',
                trainable=self._grounded_encoder_trainable,
                initializer=paddle.nn.initializer.Constant(value=0.)))

        # transform: layer norm
        mask_trans_feat = paddle.static.nn.layer_norm(
            mask_trans_feat,
            begin_norm_axis=len(mask_trans_feat.shape) - 1,
            param_attr=paddle.ParamAttr(
                name='grounded.mask_lm_trans_layer_norm_scale',
                trainable=self._grounded_encoder_trainable,
                initializer=paddle.nn.initializer.Constant(value=1.)),
            bias_attr=paddle.ParamAttr(
                name='grounded.mask_lm_trans_layer_norm_bias',
                trainable=self._grounded_encoder_trainable,
                initializer=paddle.nn.initializer.Constant(value=0.)))

        mask_lm_out_bias_attr = paddle.ParamAttr(
            name="grounded.mask_lm_out_fc.b_0",
            trainable=self._grounded_encoder_trainable,
            initializer=paddle.nn.initializer.Constant(value=0.0))

        if self._weight_sharing:
            fc_out = paddle.matmul(
                x=mask_trans_feat,
                y=paddle.static.default_main_program().global_block().var("text.word_embedding"),
                transpose_y=True)
            fc_out += paddle.create_parameter(
                shape=[self._text_voc_size],
                dtype=self._emb_dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)
        else:
            fc_out = paddle.static.nn.fc(x=mask_trans_feat,
                                         size=self._text_voc_size,
                                         weight_attr=paddle.ParamAttr(
                                             name="grounded.mask_lm_out_fc.w_0",
                                             trainable=self._grounded_encoder_trainable,
                                             initializer=self._param_initializer),
                                         bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=fc_out, label=mask_label)
        mask_lm_loss = paddle.multiply(x=mask_lm_loss, y=with_text)
        mask_lm_loss = paddle.sum(x=mask_lm_loss, axis=1, keepdim=True)
        mask_lm_loss = paddle.multiply(x=mask_lm_loss, y=mm_loss_weight)
        mean_mask_lm_loss = paddle.mean(paddle.multiply(x=mask_lm_loss, y=lm_loss_flag))
        num_lm_loss = paddle.mean(paddle.multiply(x=lm_loss_flag, y=with_text))
        mean_mask_lm_loss = mean_mask_lm_loss / (num_lm_loss + 1e-18)
        return mean_mask_lm_loss

    def get_pretraining_cmcl_output_within_batch(self, labels, with_image, with_text, is_multimodal):
        """Get the loss & accuracy for cross modal contrastive learning"""
        self.cmcl_temperature = paddle.clip(self.cmcl_temperature, min=0.001, max=0.5)

        print('-----img_txt_infonce------')
        text_feat = paddle.slice(input=self.text_enc_out, axes=[1], starts=[0], ends=[1])
        text_feat = paddle.reshape(x=text_feat, shape=[-1, self._emb_size])
        if self._with_cmcl_projection:
            text_feat = paddle.static.nn.fc(
                x=text_feat,
                size=self._emb_size,
                activation="relu",
                weight_attr=paddle.ParamAttr(
                    name="grounded.cmcl_pooled_fc_text.w_0", initializer=self._param_initializer),
                bias_attr="grounded.cmcl_pooled_fc_text.b_0")
            text_feat = paddle.nn.functional.normalize(text_feat, axis=-1)

        img_feat = paddle.slice(input=self.vit_enc_out, axes=[1], starts=[0], ends=[1])
        img_feat = paddle.reshape(x=img_feat, shape=[-1, self._emb_size])
        if self._with_cmcl_projection:
            img_feat = paddle.static.nn.fc(
                x=img_feat,
                size=self._emb_size,
                activation="relu",
                weight_attr=paddle.ParamAttr(
                    name="grounded.cmcl_pooled_fc_image.w_0", initializer=self._param_initializer),
                bias_attr="grounded.cmcl_pooled_fc_image.b_0")
            img_feat = paddle.nn.functional.normalize(img_feat, axis=-1)

        labels = paddle.reshape(paddle.cast(labels, dtype='float32'), shape=[-1])
        # paddle.static.Print(labels, message='labels', summarize=-1)
        is_multimodal = paddle.reshape(paddle.cast(is_multimodal, dtype='float32'), shape=[-1])
        # paddle.static.Print(is_multimodal, message='is_multimodal', summarize=-1)

        """compute infoNCE loss"""
        # (batch, batch)
        all_img2txt_scores = paddle.matmul(img_feat, paddle.transpose(text_feat, perm=[1, 0]))
        all_img2txt_scores = paddle.divide(x=all_img2txt_scores, y=self.cmcl_temperature)
        # paddle.static.Print(all_img2txt_scores, message='all_img2txt_scores', summarize=-1)

        pos_labels = paddle.arange(start=0, end=paddle.shape(text_feat)[0], dtype='int64')
        # paddle.static.Print(pos_labels, message='pos_labels', summarize=-1)

        img2txt_loss, img2txt_softmax = paddle.nn.functional.softmax_with_cross_entropy(
            logits=all_img2txt_scores, label=paddle.unsqueeze(pos_labels, axis=1), return_softmax=True, axis=1)
        img2txt_loss = paddle.squeeze(img2txt_loss)
        # paddle.static.Print(img2txt_loss, message='img2txt_loss', summarize=-1)
        # paddle.static.Print(img2txt_softmax, message='img2txt_softmax', summarize=-1)

        txt2img_loss, txt2img_softmax = paddle.nn.functional.softmax_with_cross_entropy(
            logits=all_img2txt_scores, label=paddle.unsqueeze(pos_labels, axis=0), return_softmax=True, axis=0)
        txt2img_loss = paddle.squeeze(txt2img_loss)
        # paddle.static.Print(txt2img_loss, message='txt2img_loss', summarize=-1)
        # paddle.static.Print(txt2img_softmax, message='txt2img_softmax', summarize=-1)

        total_loss = (img2txt_loss + txt2img_loss) / 2.0
        total_loss = paddle.multiply(total_loss, labels)
        total_loss = paddle.multiply(total_loss, is_multimodal)
        # paddle.static.Print(total_loss, message='total_loss', summarize=-1)
        loss = paddle.mean(total_loss)

        # compute accuracy
        valid_condition = paddle.cast(is_multimodal * labels, 'bool')
        valid_img2txt_ind = paddle.fluid.layers.where(valid_condition)
        # paddle.static.Print(valid_img2txt_ind, message='valid_img2txt_ind', summarize=-1)
        res = paddle.is_empty(x=valid_img2txt_ind)

        def false_func():
            valid_img2txt_softmax = paddle.gather(x=img2txt_softmax, index=valid_img2txt_ind)
            valid_txt2img_softmax = paddle.gather(x=txt2img_softmax, index=valid_img2txt_ind)
            valid_labels = paddle.gather(x=pos_labels, index=valid_img2txt_ind)
            valid_labels = paddle.reshape(x=valid_labels, shape=[-1, 1])
            img2txt_acc = paddle.metric.accuracy(input=valid_img2txt_softmax, label=valid_labels, k=5)
            txt2img_acc = paddle.metric.accuracy(input=valid_txt2img_softmax, label=valid_labels, k=5)
            return img2txt_acc, txt2img_acc

        def true_func():
            return paddle.zeros(shape=[1], dtype='float32'), paddle.zeros(shape=[1], dtype='float32')

        img2txt_acc, txt2img_acc = paddle.static.nn.cond(res, true_fn=true_func, false_fn=false_func)
        return loss, img2txt_acc, txt2img_acc

    def get_pretraining_cmcl_output_cross_batch(self, labels, with_image, with_text, is_multimodal, batch_size=None):
        """Get the loss & accuracy for cross modal contrastive learning"""
        self.cmcl_temperature = paddle.clip(self.cmcl_temperature, min=0.001, max=0.5)

        print('-----img_txt_infonce------')
        text_feat = paddle.slice(input=self.text_enc_out, axes=[1], starts=[0], ends=[1])
        text_feat = paddle.reshape(x=text_feat, shape=[batch_size, self._emb_size])
        if self._with_cmcl_projection:
            text_feat = paddle.static.nn.fc(
                x=text_feat,
                size=self._emb_size,
                activation="relu",
                weight_attr=paddle.ParamAttr(
                    name="grounded.cmcl_pooled_fc_text.w_0", initializer=self._param_initializer),
                bias_attr="grounded.cmcl_pooled_fc_text.b_0")
            text_feat = paddle.nn.functional.normalize(text_feat, axis=-1)

        img_feat = paddle.slice(input=self.vit_enc_out, axes=[1], starts=[0], ends=[1])
        img_feat = paddle.reshape(x=img_feat, shape=[batch_size, self._emb_size])
        if self._with_cmcl_projection:
            img_feat = paddle.static.nn.fc(
                x=img_feat,
                size=self._emb_size,
                activation="relu",
                weight_attr=paddle.ParamAttr(
                    name="grounded.cmcl_pooled_fc_image.w_0", initializer=self._param_initializer),
                bias_attr="grounded.cmcl_pooled_fc_image.b_0")
            img_feat = paddle.nn.functional.normalize(img_feat, axis=-1)

        worker_num = fleet.worker_num()
        print("worker num is: {}".format(fleet.worker_num()))
        print("worker index is: {}".format(fleet.worker_index()))
        # obtain cross batch data
        tot_text_feat = paddle.fluid.layers.collective._c_allgather(
            text_feat, fleet.worker_num(), use_calc_stream=True)  # (fake_tot_batch x self._emb_size)
        tot_img_feat = paddle.fluid.layers.collective._c_allgather(
            img_feat, fleet.worker_num(), use_calc_stream=True)  # (fake_tot_batch x self._emb_size)
        tot_labels = paddle.fluid.layers.collective._c_allgather(
            labels, fleet.worker_num(), use_calc_stream=True)  # (fake_tot_batch x 1)
        tot_is_multimodal = paddle.fluid.layers.collective._c_allgather(
            is_multimodal, fleet.worker_num(), use_calc_stream=True)  # (fake_tot_batch x 1)

        tot_labels = paddle.reshape(paddle.cast(tot_labels, dtype='float32'),
                                    shape=[batch_size * worker_num])
        # paddle.static.Print(tot_labels, message='tot_labels', summarize=-1)
        tot_is_multimodal = paddle.reshape(paddle.cast(tot_is_multimodal, dtype='float32'),
                                           shape=[batch_size * worker_num])
        # paddle.static.Print(tot_is_multimodal, message='tot_is_multimodal', summarize=-1)

        """compute infoNCE loss"""
        # (batch, batch)
        all_img2txt_scores = paddle.matmul(tot_img_feat, paddle.transpose(tot_text_feat, perm=[1, 0]))
        all_img2txt_scores = paddle.divide(x=all_img2txt_scores, y=self.cmcl_temperature)
        # paddle.static.Print(all_img2txt_scores, message='all_img2txt_scores', summarize=-1)

        pos_labels = paddle.arange(start=0, end=(batch_size * worker_num), dtype='int64')
        # paddle.static.Print(pos_labels, message='pos_labels', summarize=-1)

        img2txt_loss, img2txt_softmax = paddle.nn.functional.softmax_with_cross_entropy(
            logits=all_img2txt_scores, label=paddle.unsqueeze(pos_labels, axis=1), return_softmax=True, axis=1)
        img2txt_loss = paddle.squeeze(img2txt_loss)
        # paddle.static.Print(img2txt_loss, message='img2txt_loss', summarize=-1)
        # paddle.static.Print(img2txt_softmax, message='img2txt_softmax', summarize=-1)

        txt2img_loss, txt2img_softmax = paddle.nn.functional.softmax_with_cross_entropy(
            logits=all_img2txt_scores, label=paddle.unsqueeze(pos_labels, axis=0), return_softmax=True, axis=0)
        txt2img_loss = paddle.squeeze(txt2img_loss)
        # paddle.static.Print(txt2img_loss, message='txt2img_loss', summarize=-1)
        # paddle.static.Print(txt2img_softmax, message='txt2img_softmax', summarize=-1)

        total_loss = (img2txt_loss + txt2img_loss) / 2.0
        total_loss = paddle.multiply(total_loss, tot_labels)
        total_loss = paddle.multiply(total_loss, tot_is_multimodal)
        # paddle.static.Print(total_loss, message='total_loss', summarize=-1)
        loss = paddle.mean(total_loss)

        # compute accuracy
        valid_condition = paddle.cast(tot_is_multimodal * tot_labels, 'bool')
        valid_img2txt_ind = paddle.fluid.layers.where(valid_condition)
        # paddle.static.Print(valid_img2txt_ind, message='valid_img2txt_ind', summarize=-1)
        res = paddle.is_empty(x=valid_img2txt_ind)

        def false_func():
            valid_img2txt_softmax = paddle.gather(x=img2txt_softmax, index=valid_img2txt_ind)
            valid_txt2img_softmax = paddle.gather(x=txt2img_softmax, index=valid_img2txt_ind)
            valid_labels = paddle.gather(x=pos_labels, index=valid_img2txt_ind)
            valid_labels = paddle.reshape(x=valid_labels, shape=[-1, 1])
            img2txt_acc = paddle.metric.accuracy(input=valid_img2txt_softmax, label=valid_labels, k=5)
            txt2img_acc = paddle.metric.accuracy(input=valid_txt2img_softmax, label=valid_labels, k=5)
            return img2txt_acc, txt2img_acc

        def true_func():
            return paddle.zeros(shape=[1], dtype='float32'), paddle.zeros(shape=[1], dtype='float32')

        img2txt_acc, txt2img_acc = paddle.static.nn.cond(res, true_fn=true_func, false_fn=false_func)
        return loss, img2txt_acc, txt2img_acc

    def get_cmcl_scores(self):
        """Get the score for img-text pairs"""
        text_feat = paddle.slice(input=self.text_enc_out, axes=[1], starts=[0], ends=[1])
        text_feat = paddle.reshape(x=text_feat, shape=[-1, self._emb_size])
        if self._with_cmcl_projection:
            text_feat = paddle.static.nn.fc(
                x=text_feat,
                size=self._emb_size,
                activation="relu",
                weight_attr=paddle.ParamAttr(
                    name="grounded.cmcl_pooled_fc_text.w_0", initializer=self._param_initializer),
                bias_attr="grounded.cmcl_pooled_fc_text.b_0")
            text_feat = paddle.nn.functional.normalize(text_feat, axis=-1)

        img_feat = paddle.slice(input=self.vit_enc_out, axes=[1], starts=[0], ends=[1])
        img_feat = paddle.reshape(x=img_feat, shape=[-1, self._emb_size])
        if self._with_cmcl_projection:
            img_feat = paddle.static.nn.fc(
                x=img_feat,
                size=self._emb_size,
                activation="relu",
                weight_attr=paddle.ParamAttr(
                    name="grounded.cmcl_pooled_fc_image.w_0", initializer=self._param_initializer),
                bias_attr="grounded.cmcl_pooled_fc_image.b_0")
        img_feat = paddle.nn.functional.normalize(img_feat, axis=-1)

        """compute matching score"""
        # (batch, 1)
        img2txt_scores = paddle.sum(img_feat * text_feat, axis=1, keepdim=True)
        return img2txt_scores
