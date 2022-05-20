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
File: unimo_grounded.py
Author: liwei(liwei85@baidu.com)
Date: 2021-09-23 15:41
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
from model.transformer_encoder import pre_process_layer as grounded_pre_process_layer
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
                 alpha=1.0,
                 beta=0.25,
                 grounding_method='topk',
                 topk_value=100,
                 with_grounding_projection=False,
                 with_cmcl_projection=False,
                 cmcl_share_parameters=False,
                 with_grounding_mask=False,
                 grounding_mask_ratio=0.15,
                 with_grounding_pos=False,
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
        self._alpha = alpha
        self._beta = beta
        self._grounding_method = grounding_method
        self._K = topk_value
        self._grounded_encoder_trainable = grounded_encoder_trainable
        self._vit_encoder_trainable = vit_encoder_trainable
        self._text_encoder_trainable = text_encoder_trainable
        self._with_grounding_projection = with_grounding_projection
        self._cmcl_share_parameters = cmcl_share_parameters
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

        self.with_grounding_mask = with_grounding_mask
        self.grounding_mask_ratio = grounding_mask_ratio
        self.with_grounding_pos = with_grounding_pos

        if text_enc_layers is None:
            text_enc_layers = list(range(self._text_n_layer))
        if grounding_enc_layers is None:
            grounding_enc_layers = list(range(self._n_layer))

        print("text_enc_layers", text_enc_layers)
        print("grounding_enc_layers", grounding_enc_layers)
        self.text_enc_layers = text_enc_layers
        self.grounding_enc_layers = grounding_enc_layers

        # vector codebook
        self.vq_emb = paddle.static.create_parameter(
            shape=[self.num_codebook, self._emb_size],
            dtype=self._emb_dtype,
            attr=paddle.ParamAttr(
                name="vq_emb",
                initializer=paddle.fluid.initializer.UniformInitializer(
                    low=-1 / self.num_codebook, high=1 / self.num_codebook)))

        self.cmcl_temperature = paddle.static.create_parameter(
            shape=[1],
            dtype=self._emb_dtype,
            attr=paddle.ParamAttr(
                name="cmcl_temperature",
                initializer=paddle.fluid.initializer.ConstantInitializer(value=0.07)))

        self.topk_temperature = paddle.static.create_parameter(
            shape=[1],
            dtype=self._emb_dtype,
            attr=paddle.ParamAttr(
                name="topk_temperature",
                initializer=paddle.fluid.initializer.ConstantInitializer(value=100)))

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
            self._enc_v_out, self._enc_g_out, self.all_checkpoints = self.encode(image_input=image_input,
                                                                                 image_mask=image_mask,
                                                                                 gather_idx=gather_idx)
        elif text_input is not None and image_input is None:  # for textual tasks
            self._enc_g_out, self._enc_l_out, self.all_checkpoints = self.encode(text_input=text_input,
                                                                                 text_mask=text_mask,
                                                                                 gather_idx=gather_idx)
        else:  # for multi-modal tasks
            self._enc_v_out, self._enc_g_out, self._enc_l_out, self.all_checkpoints = \
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
            emb_v_out, emb_g_out, v_seq_len, g_seq_len, n_head_self_attn_mask, _checkpoints = \
                self._gen_input(image_input=image_input, image_mask=image_mask, gather_idx=gather_idx)
            all_checkpoints.extend(_checkpoints)

            if self.with_grounding_mask:
                grounded_seq_embed = paddle.reshape(emb_g_out, shape=[-1, self._emb_size])
                grounded_seq_len = paddle.shape(grounded_seq_embed)[0]
                probs = paddle.full(shape=[grounded_seq_len], fill_value=self.grounding_mask_ratio, dtype="float32")
                is_mask = paddle.bernoulli(probs)
                masked_grounded_seq_embed = grounded_seq_embed * (1 - paddle.unsqueeze(is_mask, axis=-1))
                emb_g_out = paddle.reshape(masked_grounded_seq_embed, shape=[-1, g_seq_len, self._emb_size])
                is_mask_condition = paddle.cast(is_mask, 'bool')
                mask_pos = paddle.fluid.layers.where(is_mask_condition)
                self.grounding_mask_pos = paddle.cast(x=mask_pos, dtype='int32')

            if self.with_grounding_pos:
                pos_range = paddle.unsqueeze(paddle.arange(start=2, end=g_seq_len + 2, dtype='int64'), axis=0)
                # (batch_size, g_seq_len, 1)
                batch_pos_range = paddle.unsqueeze(
                    paddle.tile(pos_range, repeat_times=[paddle.shape(emb_g_out)[0], 1]), axis=-1)
                g_pos_emb = paddle.static.nn.embedding(
                    input=batch_pos_range,
                    size=[self._text_emb_vocab_size["text.pos_embedding"], self._emb_size],
                    dtype=self._emb_dtype,
                    param_attr=paddle.ParamAttr(
                        name='text.pos_embedding',
                        trainable=self._text_encoder_trainable,
                        initializer=self._param_initializer))
                g_pos_emb = paddle.squeeze(g_pos_emb, axis=2)  # (batch_size, seq_len, emb_dim)
                emb_g_out = emb_g_out + g_pos_emb

            emb_g_out = grounded_pre_process_layer(
                emb_g_out, 'nd', self._prepostprocess_dropout,
                name="grounded.pre_encoder", trainable=self._grounded_encoder_trainable)
            emb_vg_out = paddle.concat([emb_v_out, emb_g_out], axis=1)

            enc_vg_out, grounding_checkpoints = grounded_encoder(
                enc_input=emb_vg_out,
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
                input=enc_vg_out, axes=[1], starts=[0], ends=[v_seq_len])
            enc_g_out = paddle.slice(
                input=enc_vg_out, axes=[1], starts=[v_seq_len], ends=[v_seq_len + g_seq_len])

            return enc_v_out, enc_g_out, all_checkpoints

        elif image_input is None and text_input is not None:  # for textual task
            if decoding_step:  # for step-by-step textual or img2txt generation during decoding
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
                emb_g_out, emb_l_out, g_seq_len, l_seq_len, n_head_self_attn_mask, _checkpoints = \
                    self._gen_input(text_input=text_input, text_mask=text_mask, gather_idx=gather_idx)
                all_checkpoints.extend(_checkpoints)

                if self.with_grounding_mask:
                    grounded_seq_embed = paddle.reshape(emb_g_out, shape=[-1, self._emb_size])
                    grounded_seq_len = paddle.shape(grounded_seq_embed)[0]
                    probs = paddle.full(shape=[grounded_seq_len], fill_value=self.grounding_mask_ratio, dtype="float32")
                    is_mask = paddle.bernoulli(probs)
                    masked_grounded_seq_embed = grounded_seq_embed * (1 - paddle.unsqueeze(is_mask, axis=-1))
                    emb_g_out = paddle.reshape(masked_grounded_seq_embed, shape=[-1, g_seq_len, self._emb_size])
                    is_mask_condition = paddle.cast(is_mask, 'bool')
                    mask_pos = paddle.fluid.layers.where(is_mask_condition)
                    self.grounding_mask_pos = paddle.cast(x=mask_pos, dtype='int32')

                if self.with_grounding_pos:
                    pos_range = paddle.unsqueeze(paddle.arange(start=2, end=g_seq_len + 2, dtype='int64'), axis=0)
                    # (batch_size, g_seq_len, 1)
                    batch_pos_range = paddle.unsqueeze(
                        paddle.tile(pos_range, repeat_times=[paddle.shape(emb_g_out)[0], 1]), axis=-1)
                    g_pos_emb = paddle.static.nn.embedding(
                        input=batch_pos_range,
                        size=[self._text_emb_vocab_size["text.pos_embedding"], self._emb_size],
                        dtype=self._emb_dtype,
                        param_attr=paddle.ParamAttr(
                            name='text.pos_embedding',
                            trainable=self._text_encoder_trainable,
                            initializer=self._param_initializer))
                    g_pos_emb = paddle.squeeze(g_pos_emb, axis=2)  # (batch_size, seq_len, emb_dim)
                    emb_g_out = emb_g_out + g_pos_emb

                emb_g_out = grounded_pre_process_layer(
                    emb_g_out, 'nd', self._prepostprocess_dropout,
                    name="grounded.pre_encoder", trainable=self._grounded_encoder_trainable)
                emb_gl_out = paddle.concat([emb_g_out, emb_l_out], axis=1)

                enc_gl_out, grounding_checkpoints = grounded_encoder(
                    enc_input=emb_gl_out,
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

                enc_g_out = paddle.slice(
                    input=enc_gl_out, axes=[1], starts=[0], ends=[g_seq_len])
                enc_l_out = paddle.slice(
                    input=enc_gl_out, axes=[1], starts=[g_seq_len], ends=[g_seq_len + l_seq_len])

                return enc_g_out, enc_l_out, all_checkpoints

        elif image_input is not None and text_input is not None:  # for multi-modal task
            emb_v_out, emb_g_out, emb_l_out, v_seq_len, g_seq_len, l_seq_len, n_head_self_attn_mask, _checkpoints = \
                self._gen_input(image_input=image_input, image_mask=image_mask,
                                text_input=text_input, text_mask=text_mask, gather_idx=gather_idx)
            all_checkpoints.extend(_checkpoints)

            if self.with_grounding_mask:
                grounded_seq_embed = paddle.reshape(emb_g_out, shape=[-1, self._emb_size])
                grounded_seq_len = paddle.shape(grounded_seq_embed)[0]
                probs = paddle.full(shape=[grounded_seq_len], fill_value=self.grounding_mask_ratio, dtype="float32")
                is_mask = paddle.bernoulli(probs)
                masked_grounded_seq_embed = grounded_seq_embed * (1 - paddle.unsqueeze(is_mask, axis=-1))
                emb_g_out = paddle.reshape(masked_grounded_seq_embed, shape=[-1, g_seq_len, self._emb_size])
                is_mask_condition = paddle.cast(is_mask, 'bool')
                mask_pos = paddle.fluid.layers.where(is_mask_condition)
                self.grounding_mask_pos = paddle.cast(x=mask_pos, dtype='int32')

            if self.with_grounding_pos:
                pos_range = paddle.unsqueeze(paddle.arange(start=2, end=g_seq_len + 2, dtype='int64'), axis=0)
                # (batch_size, g_seq_len, 1)
                batch_pos_range = paddle.unsqueeze(
                    paddle.tile(pos_range, repeat_times=[paddle.shape(emb_g_out)[0], 1]), axis=-1)
                g_pos_emb = paddle.static.nn.embedding(
                    input=batch_pos_range,
                    size=[self._text_emb_vocab_size["text.pos_embedding"], self._emb_size],
                    dtype=self._emb_dtype,
                    param_attr=paddle.ParamAttr(
                        name='text.pos_embedding',
                        trainable=self._text_encoder_trainable,
                        initializer=self._param_initializer))
                g_pos_emb = paddle.squeeze(g_pos_emb, axis=2)  # (batch_size, seq_len, emb_dim)
                emb_g_out = emb_g_out + g_pos_emb

            emb_g_out = grounded_pre_process_layer(
                emb_g_out, 'nd', self._prepostprocess_dropout,
                name="grounded.pre_encoder", trainable=self._grounded_encoder_trainable)
            emb_vgl_out = paddle.concat([emb_v_out, emb_g_out, emb_l_out], axis=1)

            enc_vgl_out, grounding_checkpoints = grounded_encoder(
                enc_input=emb_vgl_out,
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
                input=enc_vgl_out, axes=[1], starts=[0], ends=[v_seq_len])
            enc_g_out = paddle.slice(
                input=enc_vgl_out, axes=[1], starts=[v_seq_len], ends=[v_seq_len + g_seq_len])
            enc_l_out = paddle.slice(
                input=enc_vgl_out, axes=[1], starts=[v_seq_len + g_seq_len], ends=[v_seq_len + g_seq_len + l_seq_len])
            return enc_v_out, enc_g_out, enc_l_out, all_checkpoints
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

        return vit_enc_out, vit_seq_len, image_self_attn_mask, checkpoints

    def text_encode(self, text_input, text_mask, gather_idx=None, decoding_step=False):
        assert text_mask is not None, "text_mask should not be none"
        if decoding_step or self._is_img2txt_task:  # for img2txt tasks or step-by-step decoding
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
        return text_enc_out, text_seq_len, text_self_attn_mask, checkpoints

    def vector_quantizer(self, inputs):
        """inputs: (batch_size, seq_len, emb_dim)"""
        input_shape = paddle.shape(inputs)
        # Flatten input (batch_size * seq_len, emb_dim)
        flat_input = paddle.reshape(inputs, shape=[-1, self._emb_size])

        # Calculate distances (batch_size * seq_len, num_codebook)
        distances = (paddle.sum(paddle.pow(flat_input, 2), axis=1, keepdim=True)
                     + paddle.unsqueeze(x=paddle.sum(paddle.pow(self.vq_emb, 2), axis=1), axis=0)
                     - 2 * paddle.matmul(flat_input, paddle.transpose(self.vq_emb, perm=[1, 0])))

        # Encoding (batch_size * seq_len, 1)
        encoding_indices = paddle.unsqueeze(x=paddle.argmin(distances, axis=1), axis=1)
        # paddle.static.Print(encoding_indices, message="encoding_indices", summarize=1000)
        size_range = paddle.unsqueeze(x=paddle.arange(paddle.shape(encoding_indices)[0], dtype='int64'), axis=1)
        # (batch_size * seq_len, 2)
        index = paddle.concat([size_range, encoding_indices], axis=1)

        # (batch_size * seq_len, num_codebook)
        out_shape = [paddle.shape(encoding_indices)[0], self.num_codebook]
        # (batch_size * seq_len)
        updates = paddle.ones(shape=[paddle.shape(encoding_indices)[0]], dtype='float32')
        # (batch_size * seq_len, num_codebook)
        encodings = paddle.scatter_nd(index=index, updates=updates, shape=out_shape)
        # paddle.static.Print(encodings, message="encodings", summarize=1000)

        # Quantize and unflatten (batch_size, seq_len, emb_dim)
        quantized = paddle.reshape(x=paddle.matmul(encodings, self.vq_emb),
                                   shape=[input_shape[0], input_shape[1], self._emb_size])
        # paddle.static.Print(quantized, message="quantized", summarize=1000)
        encoding_indices_reshaped = paddle.reshape(encoding_indices, shape=[input_shape[0], input_shape[1]])

        constant_zeros = paddle.zeros_like(x=inputs)
        quantized_detach = paddle.add(constant_zeros, quantized)
        quantized_detach.stop_gradient = True
        inputs_detach = paddle.add(constant_zeros, inputs)
        inputs_detach.stop_gradient = True

        # Loss
        e_latent_loss = paddle.nn.functional.mse_loss(input=quantized_detach, label=inputs)
        # paddle.static.Print(e_latent_loss, message="e_latent_loss", summarize=1000)
        q_latent_loss = paddle.nn.functional.mse_loss(input=quantized, label=inputs_detach)
        # paddle.static.Print(q_latent_loss, message="q_latent_loss", summarize=1000)
        loss = self._alpha * q_latent_loss + self._beta * e_latent_loss

        # Straight Through Estimator
        # sg_quantized = paddle.subtract(x=quantized, y=inputs)
        # sg_quantized.stop_gradient = True
        # quantized_emb = paddle.add(x=inputs, y=sg_quantized)
        # paddle.static.Print(quantized_emb, message="quantized_emb", summarize=1000)
        return loss, quantized, encoding_indices_reshaped

    def topk_vector_quantizer(self, inputs, K=100):
        """inputs: (batch_size, seq_len, emb_dim)"""
        self.topk_temperature = paddle.clip(self.topk_temperature, min=1.0, max=10000.0)

        input_shape = paddle.shape(inputs)
        # Flatten input (batch_size * seq_len, emb_dim)
        flat_input = paddle.reshape(inputs, shape=[-1, self._emb_size])

        # Calculate distances (batch_size * seq_len, num_codebook)
        # distances = (paddle.sum(paddle.pow(flat_input, 2), axis=1, keepdim=True)
        #              + paddle.unsqueeze(x=paddle.sum(paddle.pow(self.vq_emb, 2), axis=1), axis=0)
        #              - 2 * paddle.matmul(flat_input, paddle.transpose(self.vq_emb, perm=[1, 0])))

        distances = paddle.nn.functional.sigmoid(self.topk_temperature * \
                                                 paddle.matmul(flat_input, paddle.transpose(self.vq_emb, perm=[1, 0])))
        # paddle.static.Print(distances, message="distances", summarize=2048)

        # (batch_size, num_codebook)
        cumulated_score = paddle.sum(
            paddle.reshape(distances, shape=[input_shape[0], input_shape[1], self.num_codebook]), axis=1)
        # paddle.static.Print(cumulated_score, message="cumulated_score", summarize=2048)

        # (batch_size, K)
        topk_value, topk_indices = paddle.topk(x=cumulated_score, k=K, axis=1, largest=True)
        topk_indices = paddle.reshape(topk_indices, shape=[-1, 1])
        # paddle.static.Print(topk_indices, message="topk_indices", summarize=100)

        size_range = paddle.unsqueeze(x=paddle.arange(paddle.shape(topk_indices)[0], dtype='int64'), axis=1)
        # (batch_size * K, 2)
        index = paddle.concat([size_range, topk_indices], axis=1)
        # paddle.static.Print(index, message="index", summarize=2000)

        # (batch_size * K, num_codebook)
        out_shape = [paddle.shape(topk_indices)[0], self.num_codebook]
        # (batch_size * K)
        updates = paddle.ones(shape=[paddle.shape(topk_indices)[0]], dtype='float32')
        # (batch_size * K, num_codebook)
        encodings = paddle.scatter_nd(index=index, updates=updates, shape=out_shape)
        # paddle.static.Print(encodings, message="encodings", summarize=2048)

        # Quantize and unflatten (batch_size, K, emb_dim)
        quantized = paddle.reshape(x=paddle.matmul(encodings, self.vq_emb),
                                   shape=[input_shape[0], K, self._emb_size])
        # paddle.static.Print(quantized, message="quantized", summarize=1000)
        encoding_indices_reshaped = paddle.reshape(topk_indices, shape=[input_shape[0], K])

        quantized_constant_zeros = paddle.zeros_like(x=quantized)
        quantized_detach = paddle.add(quantized_constant_zeros, quantized)
        quantized_detach.stop_gradient = True
        input_constant_zeros = paddle.zeros_like(x=inputs)
        inputs_detach = paddle.add(input_constant_zeros, inputs)
        inputs_detach.stop_gradient = True

        # (batch_size*num_codebook, seq_len)
        distances_trans = paddle.reshape(
            paddle.transpose(paddle.reshape(distances, shape=[input_shape[0], input_shape[1], self.num_codebook]),
                             perm=[0, 2, 1]),
            shape=[-1, input_shape[1]])
        # (batch_size, 1)
        batchsize_range = paddle.unsqueeze(x=paddle.arange(input_shape[0], dtype='int64'), axis=1) * self.num_codebook
        # paddle.static.Print(batchsize_range, message="batchsize_range", summarize=-1)
        # (batch_size, K)
        topk_indices = paddle.reshape(topk_indices, shape=[-1, K])
        # (batch_size * K)
        gather_index = paddle.reshape(x=(batchsize_range + topk_indices), shape=[-1])
        # paddle.static.Print(gather_index, message="gather_index", summarize=2000)
        # (batch_size * K, seq_len)
        distance_gather = paddle.gather(distances_trans, index=gather_index)
        # paddle.static.Print(distance_gather, message="distance_gather", summarize=245)
        # (batch_size, K, seq_len)
        distance_gather = paddle.reshape(distance_gather, shape=[input_shape[0], K, input_shape[1]])

        # (batch_size, K, seq_len)
        e_distance_gather_normalized = paddle.divide(
            x=distance_gather,
            y=paddle.reshape(topk_value + 1e-18, shape=[input_shape[0], K, 1]))
        # paddle.static.Print(e_distance_gather_normalized, message="e_distance_gather_normalized", summarize=245)

        # (batch_size, K, emb_dim)
        e_latent_value = paddle.matmul(e_distance_gather_normalized, inputs)
        e_latent_loss = paddle.nn.functional.mse_loss(input=quantized_detach, label=e_latent_value)
        # paddle.static.Print(e_latent_loss, message="e_latent_loss", summarize=1000)

        # (batch_size, seq_len, K)
        q_distance_gather_normailzed = paddle.divide(
            x=paddle.transpose(x=distance_gather, perm=[0, 2, 1]),
            y=paddle.unsqueeze(paddle.sum(x=distance_gather, axis=1) + 1e-18, axis=2))
        # paddle.static.Print(q_distance_gather_normailzed, message="q_distance_gather_normailzed", summarize=100)

        # (batch_size, seq_len, emb_dim)
        q_latent_value = paddle.matmul(q_distance_gather_normailzed, quantized)
        q_latent_loss = paddle.nn.functional.mse_loss(input=inputs_detach, label=q_latent_value)
        # paddle.static.Print(q_latent_loss, message="q_latent_loss", summarize=1000)

        loss = self._alpha * q_latent_loss + self._beta * e_latent_loss
        return loss, quantized, encoding_indices_reshaped

    def _gen_input(self, text_input=None, text_mask=None,
                   image_input=None, image_mask=None, gather_idx=None, decoding_step=False,
                   grounded_decoding_mask=None):
        """encode images and texts independently by Vit and RoBERTa, get the optimal grounded tokens"""
        _checkpoints = []

        if image_input is not None:
            # visual part
            self.vit_enc_out, vit_seq_len, image_self_attn_mask, vit_checkpoints = \
                self.vit_encode(image_input, image_mask)
            _checkpoints.extend(vit_checkpoints)

            # (batch_size, 1, emb_dim)
            patch_cls_embeddings = paddle.slice(input=self.vit_enc_out, axes=[1], starts=[0], ends=[1])
            # (batch_size, patch_seq_len, emb_dim)
            patch_embeddings = paddle.slice(input=self.vit_enc_out, axes=[1], starts=[1], ends=[vit_seq_len])

            if self._with_grounding_projection:
                patch_embeddings = paddle.static.nn.fc(
                    x=patch_embeddings,
                    size=self._emb_size,
                    num_flatten_dims=2,
                    activation="relu",
                    weight_attr=paddle.ParamAttr(
                        name="grounded.grounding_patch_projection.w_0",
                        initializer=self._param_initializer),
                    bias_attr="grounded.grounding_patch_projection.b_0")
                patch_embeddings = paddle.nn.functional.normalize(patch_embeddings, axis=-1)

        if text_input is not None:
            # textual part
            self.text_enc_out, text_seq_len, text_self_attn_mask, text_checkpoints = \
                self.text_encode(text_input, text_mask, gather_idx=gather_idx, decoding_step=decoding_step)
            _checkpoints.extend(text_checkpoints)

            # (batch_size, 1, emb_dim)
            text_cls_embeddings = paddle.slice(input=self.text_enc_out, axes=[1], starts=[0], ends=[1])
            # (batch_size, text_seq_len - 1, emb_dim)
            token_embeddings = paddle.slice(input=self.text_enc_out, axes=[1], starts=[1], ends=[text_seq_len])

            if self._with_grounding_projection:
                token_embeddings = paddle.static.nn.fc(
                    x=token_embeddings,
                    size=self._emb_size,
                    num_flatten_dims=2,
                    activation="relu",
                    weight_attr=paddle.ParamAttr(
                        name="grounded.grounding_token_projection.w_0",
                        initializer=self._param_initializer),
                    bias_attr="grounded.grounding_token_projection.b_0")
                token_embeddings = paddle.nn.functional.normalize(token_embeddings, axis=-1)

        if image_input is not None and text_input is not None:
            vit_text_enc_out = paddle.concat([patch_embeddings, token_embeddings], axis=1)

            if self._grounding_method == "normal":
                self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.vector_quantizer(vit_text_enc_out)
            elif self._grounding_method == "topk":
                self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.topk_vector_quantizer(vit_text_enc_out, self._K)
            elif self._grounding_method == "optimal":
                self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.vector_quantizer(vit_text_enc_out)
            else:
                raise ValueError("%s is not supported!!!" % self._grounding_method)

            batch_size = paddle.shape(self.grounded_enc_out)[0]
            g_seq_len = paddle.shape(self.grounded_enc_out)[1]

            # (batch_size, 1, g_seq_len)
            grounded_mask = paddle.ones(shape=[batch_size, 1, g_seq_len])
            # (batch_size, g_seq_len, g_seq_len)
            grounded_self_attn_mask = paddle.matmul(paddle.transpose(grounded_mask, perm=[0, 2, 1]), grounded_mask)

            # (batch_size, v_seq_len, g_seq_len)
            img_grounded_attn_mask = paddle.matmul(paddle.transpose(image_mask, perm=[0, 2, 1]), grounded_mask)
            # (batch_size, v_seq_len, l_seq_len)
            img_text_attn_mask = paddle.zeros(shape=[batch_size, vit_seq_len, text_seq_len], dtype='float32')

            grounded_img_attn_mask = paddle.matmul(paddle.transpose(grounded_mask, perm=[0, 2, 1]), image_mask)
            grounded_text_attn_mask = paddle.matmul(paddle.transpose(grounded_mask, perm=[0, 2, 1]), text_mask)

            text_img_attn_mask = paddle.zeros(shape=[batch_size, text_seq_len, vit_seq_len], dtype='float32')
            text_grounded_attn_mask = paddle.matmul(paddle.transpose(text_mask, perm=[0, 2, 1]), grounded_mask)

            image_row = paddle.concat([image_self_attn_mask, img_grounded_attn_mask, img_text_attn_mask], axis=2)
            grounded_row = paddle.concat([grounded_img_attn_mask, grounded_self_attn_mask, grounded_text_attn_mask],
                                         axis=2)
            text_row = paddle.concat([text_img_attn_mask, text_grounded_attn_mask, text_self_attn_mask], axis=2)
            vgl_self_attn_mask = paddle.concat([image_row, grounded_row, text_row], axis=1)
            vgl_self_attn_mask = paddle.scale(
                x=vgl_self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
            n_head_self_attn_mask = paddle.stack(
                x=[vgl_self_attn_mask] * self._n_head, axis=1)
            n_head_self_attn_mask.stop_gradient = True

            return self.vit_enc_out, self.grounded_enc_out, self.text_enc_out, \
                   vit_seq_len, g_seq_len, text_seq_len, n_head_self_attn_mask, _checkpoints

        elif image_input is not None and text_input is None:
            if self._grounding_method == "normal":
                self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.vector_quantizer(patch_embeddings)
            elif self._grounding_method == "topk":
                self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.topk_vector_quantizer(patch_embeddings, self._K)
            elif self._grounding_method == "optimal":
                self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.vector_quantizer(patch_embeddings)
            else:
                raise ValueError("%s is not supported!!!" % self._grounding_method)

            batch_size = paddle.shape(self.grounded_enc_out)[0]
            g_seq_len = paddle.shape(self.grounded_enc_out)[1]

            # (batch_size, 1, g_seq_len)
            grounded_mask = paddle.ones(shape=[batch_size, 1, g_seq_len])
            vg_mask = paddle.concat([image_mask, grounded_mask], axis=2)
            vg_self_attn_mask = paddle.matmul(paddle.transpose(vg_mask, perm=[0, 2, 1]), vg_mask)
            vg_self_attn_mask = paddle.scale(
                x=vg_self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
            n_head_self_attn_mask = paddle.stack(
                x=[vg_self_attn_mask] * self._n_head, axis=1)
            n_head_self_attn_mask.stop_gradient = True

            return self.vit_enc_out, self.grounded_enc_out, vit_seq_len, g_seq_len, n_head_self_attn_mask, _checkpoints

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
                if self._grounding_method == "normal":
                    self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                        self.vector_quantizer(token_embeddings)
                elif self._grounding_method == "topk":
                    self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                        self.topk_vector_quantizer(token_embeddings, self._K)
                elif self._grounding_method == "optimal":
                    self.vq_loss, self.grounded_enc_out, self.grounded_encoding_indices = \
                        self.vector_quantizer(token_embeddings)
                else:
                    raise ValueError("%s is not supported!!!" % self._grounding_method)

                batch_size = paddle.shape(self.grounded_enc_out)[0]
                g_seq_len = paddle.shape(self.grounded_enc_out)[1]

                # (batch_size, 1, g_seq_len)
                grounded_mask = paddle.ones(shape=[batch_size, 1, g_seq_len])

                gl_mask = paddle.concat([grounded_mask, text_mask], axis=2)
                gl_self_attn_mask = paddle.matmul(paddle.transpose(gl_mask, perm=[0, 2, 1]), gl_mask)
                gl_self_attn_mask = paddle.scale(
                    x=gl_self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
                n_head_self_attn_mask = paddle.stack(
                    x=[gl_self_attn_mask] * self._n_head, axis=1)
                n_head_self_attn_mask.stop_gradient = True

                return self.grounded_enc_out, self.text_enc_out, g_seq_len, text_seq_len, n_head_self_attn_mask, _checkpoints

    def get_checkpoints(self):
        return self.all_checkpoints

    def get_text_encoder_output(self):
        return self.text_enc_out

    def get_vit_encoder_output(self):
        return self.vit_enc_out

    def get_grounded_output(self):
        return self.grounded_enc_out

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

    def get_img_txt_matching_output(self, labels, is_multimodal=None, num_labels=2, fusion_method="sum"):
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
            if self._cmcl_share_parameters:
                text_feat = paddle.static.nn.fc(
                    x=text_feat,
                    size=self._emb_size,
                    activation="relu",
                    weight_attr=paddle.ParamAttr(
                        name="grounded.grounding_token_projection.w_0",
                        initializer=self._param_initializer),
                    bias_attr="grounded.grounding_token_projection.b_0")
            else:
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
            if self._cmcl_share_parameters:
                img_feat = paddle.static.nn.fc(
                    x=img_feat,
                    size=self._emb_size,
                    activation="relu",
                    weight_attr=paddle.ParamAttr(
                        name="grounded.grounding_patch_projection.w_0",
                        initializer=self._param_initializer),
                    bias_attr="grounded.grounding_patch_projection.b_0")
            else:
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
            if self._cmcl_share_parameters:
                text_feat = paddle.static.nn.fc(
                    x=text_feat,
                    size=self._emb_size,
                    activation="relu",
                    weight_attr=paddle.ParamAttr(
                        name="grounded.grounding_token_projection.w_0",
                        initializer=self._param_initializer),
                    bias_attr="grounded.grounding_token_projection.b_0")
            else:
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
            if self._cmcl_share_parameters:
                img_feat = paddle.static.nn.fc(
                    x=img_feat,
                    size=self._emb_size,
                    activation="relu",
                    weight_attr=paddle.ParamAttr(
                        name="grounded.grounding_patch_projection.w_0",
                        initializer=self._param_initializer),
                    bias_attr="grounded.grounding_patch_projection.b_0")
            else:
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
            if self._cmcl_share_parameters:
                text_feat = paddle.static.nn.fc(
                    x=text_feat,
                    size=self._emb_size,
                    activation="relu",
                    weight_attr=paddle.ParamAttr(
                        name="grounded.grounding_token_projection.w_0",
                        initializer=self._param_initializer),
                    bias_attr="grounded.grounding_token_projection.b_0")
            else:
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
            if self._cmcl_share_parameters:
                img_feat = paddle.static.nn.fc(
                    x=img_feat,
                    size=self._emb_size,
                    activation="relu",
                    weight_attr=paddle.ParamAttr(
                        name="grounded.grounding_patch_projection.w_0",
                        initializer=self._param_initializer),
                    bias_attr="grounded.grounding_patch_projection.b_0")
            else:
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

    def get_pretraining_mask_grounding_output(self):
        """Get the loss & accuracy for pretraining language"""
        grounded_seq_label = paddle.reshape(self.grounded_encoding_indices, shape=[-1, 1])
        grounding_mask_label = paddle.gather(grounded_seq_label, index=self.grounding_mask_pos)

        # extract masked tokens' feature
        reshaped_grounding_feat = paddle.reshape(self._enc_g_out, shape=[-1, self._emb_size])
        mask_feat = paddle.gather(x=reshaped_grounding_feat, index=self.grounding_mask_pos)

        # transform: fc
        mask_trans_feat = paddle.static.nn.fc(
            x=mask_feat,
            size=self._emb_size,
            activation=self._hidden_act,
            weight_attr=paddle.ParamAttr(
                name='grounded.mask_g_trans_fc.w_0',
                trainable=self._grounded_encoder_trainable,
                initializer=self._param_initializer),
            bias_attr=paddle.ParamAttr(
                name='grounded.mask_g_trans_fc.b_0',
                trainable=self._grounded_encoder_trainable,
                initializer=paddle.nn.initializer.Constant(value=0.)))

        # transform: layer norm
        mask_trans_feat = paddle.static.nn.layer_norm(
            mask_trans_feat,
            begin_norm_axis=len(mask_trans_feat.shape) - 1,
            param_attr=paddle.ParamAttr(
                name='grounded.mask_g_trans_layer_norm_scale',
                trainable=self._grounded_encoder_trainable,
                initializer=paddle.nn.initializer.Constant(value=1.)),
            bias_attr=paddle.ParamAttr(
                name='grounded.mask_g_trans_layer_norm_bias',
                trainable=self._grounded_encoder_trainable,
                initializer=paddle.nn.initializer.Constant(value=0.)))

        mask_lm_out_bias_attr = paddle.ParamAttr(
            name="grounded.mask_g_out_fc.b_0",
            trainable=self._grounded_encoder_trainable,
            initializer=paddle.nn.initializer.Constant(value=0.0))

        if self._weight_sharing:
            fc_out = paddle.matmul(
                x=mask_trans_feat,
                y=paddle.static.default_main_program().global_block().var("vq_emb"),
                transpose_y=True)
            fc_out += paddle.create_parameter(
                shape=[self.num_codebook],
                dtype=self._emb_dtype,
                attr=mask_lm_out_bias_attr,
                is_bias=True)
        else:
            fc_out = paddle.static.nn.fc(x=mask_trans_feat,
                                         size=self.num_codebook,
                                         weight_attr=paddle.ParamAttr(
                                             name="grounded.mask_g_out_fc.w_0",
                                             trainable=self._grounded_encoder_trainable,
                                             initializer=self._param_initializer),
                                         bias_attr=mask_lm_out_bias_attr)

        mask_g_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=fc_out, label=grounding_mask_label)
        mask_g_loss = paddle.sum(x=mask_g_loss, axis=1, keepdim=True)
        mean_mask_g_loss = paddle.mean(mask_g_loss)
        return mean_mask_g_loss
