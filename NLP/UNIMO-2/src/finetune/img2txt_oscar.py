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
File: img2txt_oscar.py
Author: liwei(liwei85@baidu.com)
Date: 2021-10-25 16:06
Desc: img to text generation
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time
import numpy as np
import glob
import json
import codecs

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from eval.gen_eval import GenerationEval
from finetune.trigram_blocking import TrigramBlocking

from model.transformer_encoder import encoder as grounded_encoder
from model.transformer_encoder import pre_process_layer as grounded_pre_process_layer
from model.transformer_encoder import encoder as text_encoder
from model.transformer_encoder import pre_process_layer as text_pre_process_layer
from model.vision_transformer_encoder import encoder as vit_encoder
from model.vision_transformer_encoder import pre_process_layer as vit_pre_process_layer
from utils.pos_emb_interpolate import interpolate_pos_embed


class Img2Txt(object):
    def __init__(self, args, vl_config, tokenizer):
        self.vl_config = vl_config
        self.weight_sharing = args.weight_sharing
        self.max_seq_len = args.max_seq_len
        self.max_obj_len = args.max_obj_len
        self.label_smooth = args.label_smooth
        self.tgt_type_id = args.tgt_type_id
        self.tokenizer = tokenizer
        self.vocab_size = vl_config["text_vocab_size"]
        self._emb_dtype = "float32"

        # for beam_search decoding
        self.do_decode = args.do_decode
        self.length_penalty = args.length_penalty
        self.max_out_len = args.max_out_len
        self.min_out_len = args.min_out_len
        self.block_trigram = args.block_trigram
        self.beam_size = args.beam_size

        self.patch_seq_len = self.vl_config['image_size'] * self.vl_config['image_size'] // \
                             (self.vl_config['resolution'] * self.vl_config['resolution'])
        # directly utilize Conv2d to extract path and linear transforming
        self.patch_emb_size = self.vl_config['resolution'] * self.vl_config['resolution'] * 3

        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.sep_token_id
        self.evaluator = GenerationEval(args)
        self.task_type = "img2txt"

        self.model_type = args.model_type
        self.grounding_method = args.grounding_method
        self.topk_value = args.topk_value
        self.with_grounding_projection = args.with_grounding_projection
        self.with_grounding_pos = args.with_grounding_pos

        self.text_enc_layers = [int(i) for i in args.text_enc_layers.split(',')]
        self.grounding_enc_layers = [int(i) for i in args.grounding_enc_layers.split(',')]

    def cal_logit(self, enc_out, tgt_pos):
        enc_out = fluid.layers.reshape(x=enc_out,
                                       shape=[-1, self.vl_config["hidden_size"]])
        if tgt_pos:
            tgt_pos = fluid.layers.cast(x=tgt_pos, dtype='int32')
            tgt_feat = fluid.layers.gather(input=enc_out, index=tgt_pos)
        else:
            tgt_feat = enc_out

        tgt_trans_feat = fluid.layers.fc(
            input=tgt_feat,
            size=self.vl_config["hidden_size"],
            act=self.vl_config["hidden_act"],
            param_attr=fluid.ParamAttr(
                name="grounded.mask_lm_trans_fc.w_0",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="grounded.mask_lm_trans_fc.b_0",
                initializer=fluid.initializer.Constant(0.)))

        tgt_trans_feat = fluid.layers.layer_norm(
            tgt_trans_feat,
            begin_norm_axis=len(tgt_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='grounded.mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='grounded.mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(1.)))

        seq2seq_out_bias_attr = fluid.ParamAttr(
            name="grounded.mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))

        if self.weight_sharing:
            fc_out = fluid.layers.matmul(
                x=tgt_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    "text.word_embedding"),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(
                shape=[self.vl_config['text_vocab_size']],
                dtype="float32",
                attr=seq2seq_out_bias_attr,
                is_bias=True)
        else:
            out_size = self.vl_config['text_vocab_size']
            fc_out = fluid.layers.fc(input=tgt_trans_feat,
                                     size=out_size,
                                     param_attr=fluid.ParamAttr(
                                         name="grounded.mask_lm_out_fc.w_0",
                                         initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                                     bias_attr=seq2seq_out_bias_attr)

        return fc_out

    def to_tensor(self, shapes, dtypes, lod_levels):
        return [fluid.layers.data(name="placeholder_" + str(i), shape=shapes[i], dtype=dtypes[i],
                                  lod_level=lod_levels[i]) for i in range(len(shapes))]

    def create_model(self, decoding=False):
        """create model"""
        if decoding:
            return self.fast_decode()

        img_input_shapes = [[-1, self.vl_config['image_size'], self.vl_config['image_size'], 3],  # image_pixel_input
                            [-1, 1, self.patch_seq_len + 1]]  # image_mask
        img_input_dtypes = ['float32', 'float32']
        img_input_lod_levels = [0, 0]

        emb_num = 3
        text_input_shapes = [[-1, self.max_seq_len, 1]] * emb_num + \
                            [[-1, self.max_seq_len, self.max_seq_len], [-1, 1], [-1, 1]]
        text_input_dtypes = ['int64'] * emb_num + ['float32', 'int64', 'int64']
        text_input_lod_levels = [0] * emb_num + [0, 0, 0]

        shapes = img_input_shapes + text_input_shapes
        dtypes = img_input_dtypes + text_input_dtypes
        lod_levels = img_input_lod_levels + text_input_lod_levels

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)

        image_input = {}
        text_input = {}
        image_input["pixel_embedding"], image_mask, text_input["text.word_embedding"], \
        text_input["text.sent_embedding"], text_input["text.pos_embedding"], text_mask, tgt_labels, tgt_pos = inputs

        if self.model_type == "grounded":
            model = GroundingModelForImg2Txt(text_input=text_input,
                                             text_mask=text_mask,
                                             image_input=image_input,
                                             image_mask=image_mask,
                                             config=self.vl_config,
                                             weight_sharing=self.weight_sharing,
                                             grounding_method=self.grounding_method,
                                             topk_value=self.topk_value,
                                             with_grounding_projection=self.with_grounding_projection,
                                             with_grounding_pos=self.with_grounding_pos,
                                             text_enc_layers=self.text_enc_layers,
                                             grounding_enc_layers=self.grounding_enc_layers)
        elif self.model_type == "baseline":
            model = BaselineForImg2Txt(text_input=text_input,
                                       text_mask=text_mask,
                                       image_input=image_input,
                                       image_mask=image_mask,
                                       config=self.vl_config,
                                       weight_sharing=self.weight_sharing,
                                       text_enc_layers=self.text_enc_layers,
                                       grounding_enc_layers=self.grounding_enc_layers)
        else:
            raise ValueError("The model_type is invalid!!!")

        enc_out = model.get_text_sequence_output()
        fc_out = self.cal_logit(enc_out, tgt_pos)

        if self.label_smooth:
            out_size = self.vl_config['text_vocab_size']
            labels = fluid.layers.label_smooth(
                label=fluid.layers.one_hot(
                    input=tgt_labels, depth=out_size),
                epsilon=self.label_smooth)

            ce_loss = layers.softmax_with_cross_entropy(
                logits=fc_out, label=labels, soft_label=True)
        else:
            ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                logits=fc_out, label=tgt_labels, return_softmax=True)

        loss = fluid.layers.mean(x=ce_loss)
        graph_vars = {"loss": loss}
        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars, model.get_checkpoints()

    def fast_decode(self):
        input_shapes = [[-1, self.vl_config['image_size'], self.vl_config['image_size'], 3],  # image_pixel_input
                        [-1, 1, self.patch_seq_len + 1],  # image_mask
                        [-1, 1],  # image_id
                        [-1, self.max_obj_len, 1],  # padded_obj_token_id
                        [-1, self.max_obj_len, 1],  # padded_obj_sent_ids
                        [-1, self.max_obj_len, 1],  # padded_obj_pos_ids
                        [-1, self.max_obj_len, self.max_obj_len]]  # obj_mask
        input_dtypes = ['float32', 'float32', 'int32', 'int64', 'int64', 'int64', 'float32']
        input_lod_levels = [0, 0, 0, 0, 0, 0, 0]

        shapes = input_shapes + [[-1, 1, 1], [-1, 1, 1], [-1, 1], [-1], [-1, 1, self.max_obj_len]]
        dtypes = input_dtypes + ['int64', 'int64', 'float32', 'int32', 'float32']
        lod_levels = input_lod_levels + [2, 2, 2, 0, 0]

        inputs = self.to_tensor(shapes, dtypes, lod_levels)
        pyreader = fluid.io.DataLoader.from_generator(feed_list=inputs, capacity=70, iterable=False)

        image_input = {}
        obj_input = {}
        image_input["pixel_embedding"], image_mask, image_ids, \
        obj_input["text.word_embedding"], obj_input["text.sent_embedding"], obj_input["text.pos_embedding"], obj_mask,\
        tgt_ids, tgt_pos, init_scores, parent_idx, tgt_input_mask = inputs

        if self.model_type == "grounded":
            model = GroundingModelForImg2Txt(text_input=obj_input,
                                             text_mask=obj_mask,
                                             image_input=image_input,
                                             image_mask=image_mask,
                                             config=self.vl_config,
                                             weight_sharing=self.weight_sharing,
                                             decoding=True,
                                             gather_idx=parent_idx,
                                             grounding_method=self.grounding_method,
                                             topk_value=self.topk_value,
                                             with_grounding_projection=self.with_grounding_projection,
                                             with_grounding_pos=self.with_grounding_pos,
                                             text_enc_layers=self.text_enc_layers,
                                             grounding_enc_layers=self.grounding_enc_layers)
        elif self.model_type == "baseline":
            model = BaselineForImg2Txt(text_input=obj_input,
                                       text_mask=obj_mask,
                                       image_input=image_input,
                                       image_mask=image_mask,
                                       config=self.vl_config,
                                       weight_sharing=self.weight_sharing,
                                       decoding=True,
                                       gather_idx=parent_idx,
                                       text_enc_layers=self.text_enc_layers,
                                       grounding_enc_layers=self.grounding_enc_layers)
        else:
            raise ValueError("The model_type is invalid!!!")

        max_len = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=self.max_out_len, force_cpu=True)
        min_len = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=self.min_out_len, force_cpu=True)
        neg_inf = layers.fill_constant(
            shape=[1], dtype='float32', value=-1e18)
        step_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=0, force_cpu=True)
        step_next_idx = layers.fill_constant(
            shape=[1], dtype=tgt_ids.dtype, value=1, force_cpu=True)
        cond = layers.less_than(x=step_idx, y=max_len)
        while_op = layers.While(cond)

        ids = layers.array_write(layers.reshape(tgt_ids, (-1, 1)), step_idx)
        # pos_biases = layers.array_write(layers.reshape(tgt_pos, (-1, 1)), step_idx)
        pos_biases = layers.array_write(tgt_pos, step_idx)
        scores = layers.array_write(init_scores, step_idx)

        batch_size = paddle.shape(tgt_ids)[0]
        grounding_mask = paddle.ones(shape=[batch_size, 1, model.out_seq_len], dtype=self._emb_dtype)
        grounding_masks = layers.array_write(grounding_mask, step_idx)

        tgt_masks = layers.array_write(tgt_input_mask, step_idx)
        trigram_blocking = TrigramBlocking(tgt_ids, self.tokenizer, beam_size=self.beam_size)

        with while_op.block():
            pre_ids = layers.array_read(array=ids, i=step_idx)
            pre_ids = layers.reshape(pre_ids, (-1, 1, 1), inplace=True)
            pre_scores = layers.array_read(array=scores, i=step_idx)
            pos_bias = layers.array_read(array=pos_biases, i=step_idx)
            pos_bias = layers.gather(input=pos_bias, index=parent_idx)

            def gen_batch_like(value, dtype="int64", shape=[-1, 1, 1], is_scalar=True):
                if is_scalar:
                    return layers.fill_constant_batch_size_like(
                        input=parent_idx, value=value, shape=shape, dtype=dtype)
                else:
                    return layers.elementwise_mul(
                        x=layers.fill_constant_batch_size_like(
                            input=parent_idx, value=1, shape=shape, dtype=dtype),
                        y=value, axis=0)

            tmp_grounding_mask = layers.array_read(grounding_masks, i=step_idx)
            tmp_grounding_mask = layers.gather(input=tmp_grounding_mask, index=parent_idx)
            append_1_mask = gen_batch_like(1.0, dtype=tmp_grounding_mask.dtype)
            pre_grounding_mask = layers.concat([tmp_grounding_mask, append_1_mask], axis=2)

            tmp_text_mask = layers.array_read(tgt_masks, i=step_idx)
            tmp_text_mask = layers.gather(input=tmp_text_mask, index=parent_idx)
            append_1_mask = gen_batch_like(1.0, dtype=tmp_text_mask.dtype)
            pre_text_mask = layers.concat([tmp_text_mask, append_1_mask], axis=2)

            pre_pos = gen_batch_like(step_idx, is_scalar=False)
            pre_pos = pre_pos + pos_bias  ####################### pos start from 2
            pre_sent = gen_batch_like(self.tgt_type_id, dtype=pre_ids.dtype)

            dec_emb_ids = {"text.word_embedding": pre_ids, "text.sent_embedding": pre_sent,
                           "text.pos_embedding": pre_pos}
            dec_out, _ = model.encode(text_input=dec_emb_ids,
                                      text_mask=pre_text_mask,
                                      gather_idx=parent_idx,
                                      decoding_step=True,
                                      grounded_decoding_mask=pre_grounding_mask)
            fc_out = self.cal_logit(dec_out, None)

            # prevent generating end token if length less than min_out_len
            eos_index = layers.fill_constant(shape=[layers.shape(fc_out)[0]],
                                             dtype='int64',
                                             value=self.eos_id)
            eos_index = fluid.one_hot(eos_index, depth=self.vocab_size)
            less_cond = layers.cast(layers.less_than(x=step_idx, y=min_len), dtype='float32')
            less_val = layers.elementwise_mul(less_cond, neg_inf)
            eos_val = layers.elementwise_mul(eos_index, less_val, axis=0)
            revised_logits = layers.elementwise_add(fc_out, eos_val, axis=0)

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
                ((5.0 + fluid.layers.cast(step_idx, pre_scores.dtype)) / 6.0), self.length_penalty)
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
                pre_scores_wo_len_penalty = fluid.layers.elementwise_add(x=trigram_blocking.delta_score_out,
                                                                         y=pre_scores_wo_len_penalty,
                                                                         axis=0)
            # => [N, topk]
            accu_scores = layers.elementwise_add(
                x=layers.log(topk_scores), y=pre_scores_wo_len_penalty, axis=0)

            cur_timestep_length_penalty = layers.pow(((5.0 + layers.cast(step_next_idx, accu_scores.dtype)) / 6.0),
                                                     self.length_penalty)
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
                end_id=self.eos_id,
                return_parent_idx=True)

            layers.increment(x=step_idx, value=1.0, in_place=True)
            layers.increment(x=step_next_idx, value=1.0, in_place=True)
            # cell states(caches) have been updated in wrap_decoder,
            # only need to update beam search states here.
            layers.array_write(selected_ids, i=step_idx, array=ids)
            layers.array_write(selected_scores, i=step_idx, array=scores)
            layers.array_write(pre_text_mask, i=step_idx, array=tgt_masks)
            layers.array_write(pre_grounding_mask, i=step_idx, array=grounding_masks)
            layers.array_write(pos_bias, i=step_idx, array=pos_biases)
            layers.assign(gather_idx, parent_idx)

            length_cond = layers.less_than(x=step_idx, y=max_len)
            finish_cond = layers.logical_not(layers.is_empty(x=selected_ids))
            layers.logical_and(x=length_cond, y=finish_cond, out=cond)

        finished_ids, finished_scores = layers.beam_search_decode(
            ids, scores, beam_size=self.beam_size, end_id=self.eos_id)

        graph_vars = {
            "finished_ids": finished_ids,
            "finished_scores": finished_scores,
            "image_ids": image_ids
        }

        for k, v in graph_vars.items():
            v.persistable = True

        return pyreader, graph_vars

    def post_process_seq(self, seq):
        """
        Post-process the beam-search decoded sequence. Truncate from the first
        <eos> and remove the <bos> and <eos> tokens currently.
        """
        eos_pos = len(seq)
        for i, idx in enumerate(seq):
            if idx == self.eos_id:
                eos_pos = i
                break
        seq = seq[1:eos_pos]
        return seq

    def remove_special_tokens(self, seq, special_tokens):
        """Remove special tokens from output sequence"""
        seq = [idx for idx in seq if idx not in special_tokens]
        return seq

    def evaluate(self, resource, eval_phase, graph_vars, features=None,
                 output_path=None, dev_count=1, gpu_id=0):
        exe, program, pyreader = resource["exe"], resource["program"], resource["pyreader"]

        if eval_phase == "train":
            fetch_list = [graph_vars["loss"].name]
            outputs = exe.run(fetch_list=fetch_list)
            np_loss = outputs[0]
            ret = {"loss": np.mean(np_loss), "ppl": np.exp(np.mean(np_loss))}
            return ret

        if self.do_decode:
            return_numpy = False
            outfile = output_path + "/" + eval_phase
            outfile_part = outfile + ".part" + str(gpu_id)
            writer = codecs.open(outfile_part, "w", encoding='utf-8')
            fetch_keys = ["finished_ids", "finished_scores", "image_ids"]
            special_tokens = [self.tokenizer.cls_token_id,
                              self.tokenizer.sep_token_id,
                              self.tokenizer.mask_token_id,
                              self.tokenizer.pad_token_id,
                              self.tokenizer.unk_token_id]
        else:
            steps = 0
            cost = 0.0
            return_numpy = True
            fetch_keys = ["loss"]

        fetch_list = [graph_vars[key].name for key in fetch_keys]

        time_begin = time.time()
        pyreader.start()
        while True:
            try:
                outputs = exe.run(program=program,
                                  fetch_list=fetch_list,
                                  return_numpy=return_numpy)
                if not self.do_decode:
                    np_loss = outputs[0]
                    cost += np.mean(np_loss)
                    steps += 1
                else:
                    seq_ids, seq_scores, image_ids = outputs
                    seq_ids_list, seq_scores_list = [seq_ids], [seq_scores] \
                        if isinstance(seq_ids, paddle.fluid.core.LoDTensor) else (seq_ids, seq_scores)

                    image_ids = np.array(image_ids).reshape(-1).tolist()
                    data_idx = 0

                    for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):
                        # How to parse the results:
                        #   Suppose the lod of seq_ids is:
                        #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
                        #   then from lod[0]:
                        #     there are 2 source sentences, beam width is 3.
                        #   from lod[1]:
                        #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
                        #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
                        # hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
                        # scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
                        for i in range(len(seq_ids.lod()[0]) - 1):  # for each source sentence
                            start = seq_ids.lod()[0][i]
                            end = seq_ids.lod()[0][i + 1]
                            max_cand = None
                            for j in range(end - start):  # for each candidate
                                sub_start = seq_ids.lod()[1][start + j]
                                sub_end = seq_ids.lod()[1][start + j + 1]
                                token_ids = [int(idx) for idx in self.post_process_seq(
                                    np.array(seq_ids)[sub_start:sub_end])]
                                # print(len(token_ids))

                                hyp_ids = self.remove_special_tokens(token_ids, special_tokens)
                                hyp_tokens = self.tokenizer.convert_ids_to_tokens(hyp_ids)

                                hyp_str = self.tokenizer.gptbpe_tokenizer.decode(hyp_tokens)
                                # hyp_str_tokens = []
                                # for hyp_token in hyp_tokens:
                                #     hyp_str_tokens.append(self.tokenizer.gptbpe_tokenizer.decode_token(hyp_token))
                                # hyp_str = ' '.join(hyp_str_tokens)

                                hyp_str = re.sub('\\s+', ' ', hyp_str)
                                # print(hyp_str)

                                score = np.array(seq_scores)[sub_end - 1]
                                if (not max_cand) or score > max_cand[1]:
                                    max_cand = (hyp_str, score)

                            image_id = image_ids[data_idx]
                            data_idx += 1
                            pred = max_cand[0]
                            writer.write("%d\t%s\n" % (image_id, pred))

            except fluid.core.EOFException:
                pyreader.reset()
                break

        time_end = time.time()
        if not self.do_decode:
            eval_result = "loss: %f, ppl: %f" % (cost / steps, np.exp(cost / steps))
            print("[%s evaluation] %s, elapsed time: %f s"
                  % (eval_phase, eval_result, time_end - time_begin))
        else:
            writer.close()
            tmp_writer = open("%s/%s_dec_finish.%d" % (output_path, eval_phase, gpu_id), "w")
            tmp_writer.close()
            if gpu_id != 0:
                return

            while True:
                ret = os.popen('find %s -maxdepth 1 -name "%s_dec_finish.*"' %
                               (output_path, eval_phase)).readlines()
                if len(ret) != dev_count:
                    time.sleep(1)
                    continue
                else:
                    break

            all_outfiles = glob.glob("%s.part*" % outfile)
            img_caption_res = []
            unique_image_ids = []
            for cur_file in all_outfiles:
                for line in open(cur_file, 'r'):
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        print("Warning: invalid output %s " % line.strip())
                        continue

                    image_id, caption = parts

                    if image_id in unique_image_ids:
                        print("Warning: Repeated image_id %s" % str(image_id))
                        continue
                    unique_image_ids.append(image_id)
                    img_caption_res.append({"image_id": int(image_id), "caption": caption})

            fout = open(outfile, 'w')
            fout.write(json.dumps(img_caption_res))
            fout.close()
            os.system("rm %s.part*" % outfile)
            os.system("rm %s/%s_dec_finish.*" % (output_path, eval_phase))

            eval_result = self.evaluator.eval(outfile,
                                              phase=eval_phase.split("_")[0], features=features)
            print("[%s evaluation] %s, elapsed time: %f s"
                  % (eval_phase, eval_result, time_end - time_begin))


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


class GroundingModelForImg2Txt(object):
    def __init__(self,
                 image_input=None,
                 image_mask=None,
                 text_input=None,
                 text_mask=None,
                 config=None,
                 weight_sharing=True,
                 decoding=False,
                 gather_idx=None,
                 grounded_encoder_trainable=True,
                 vit_encoder_trainable=True,
                 text_encoder_trainable=True,
                 grounding_method='topk',
                 topk_value=100,
                 with_grounding_projection=False,
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
        self._grounding_method = grounding_method
        self._K = topk_value
        self._grounded_encoder_trainable = grounded_encoder_trainable
        self._vit_encoder_trainable = vit_encoder_trainable
        self._text_encoder_trainable = text_encoder_trainable
        self._with_grounding_projection = with_grounding_projection
        self.with_grounding_pos = with_grounding_pos

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = paddle.fluid.initializer.TruncatedNormalInitializer(scale=config['initializer_range'])
        self._bias_initializer = paddle.fluid.initializer.ConstantInitializer(value=0.0)

        assert text_input is not None or image_input is not None, "text_input and image_input cannot both be None"
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

        # vector codebook
        self.vq_emb = paddle.static.create_parameter(
            shape=[self.num_codebook, self._emb_size],
            dtype=self._emb_dtype,
            attr=paddle.ParamAttr(
                name="vq_emb",
                initializer=paddle.fluid.initializer.UniformInitializer(
                    low=-1 / self.num_codebook, high=1 / self.num_codebook)))

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
        if text_input is None and image_input is not None:  # for img2txt inference
            self._enc_v_out, self._enc_g_out, self.all_checkpoints, self.out_seq_len = \
                self.encode(image_input=image_input,
                            image_mask=image_mask,
                            gather_idx=gather_idx)
        elif text_input is not None and image_input is None:  # for img2txt step-by-step decoding
            self._enc_g_out, self._enc_l_out, self.all_checkpoints = self.encode(text_input=text_input,
                                                                                 text_mask=text_mask,
                                                                                 gather_idx=gather_idx)
        else:  # for img2txt training
            self._enc_v_out, self._enc_g_out, self._enc_l_out, self.all_checkpoints, self.out_seq_len = \
                self.encode(text_input=text_input,
                            text_mask=text_mask,
                            gather_idx=gather_idx,
                            image_input=image_input,
                            image_mask=image_mask)

    def encode(self, text_input=None, text_mask=None, gather_idx=None,
               image_input=None, image_mask=None, decoding_step=False, grounded_decoding_mask=None):
        all_checkpoints = []
        # padding id in vocabulary must be set to 0
        if text_input is None and image_input is not None:  # for img2txt inference
            emb_v_out, emb_g_out, v_seq_len, g_seq_len, n_head_self_attn_mask, _checkpoints = \
                self._gen_input(image_input=image_input, image_mask=image_mask, gather_idx=gather_idx)
            all_checkpoints.extend(_checkpoints)

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
                g_pos_emb = paddle.squeeze(g_pos_emb, axis=2)  # (batch_size, g_seq_len, emb_dim)
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

            vg_seq_len = v_seq_len + g_seq_len
            enc_v_out = paddle.slice(
                input=enc_vg_out, axes=[1], starts=[0], ends=[v_seq_len])
            enc_g_out = paddle.slice(
                input=enc_vg_out, axes=[1], starts=[v_seq_len], ends=[v_seq_len + g_seq_len])

            return enc_v_out, enc_g_out, all_checkpoints, vg_seq_len

        elif image_input is None and text_input is not None:  # for img2txt step-by-step decoding
            assert decoding_step is True, "decoding_step must be set True"

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

        elif image_input is not None and text_input is not None:  # for img2txt training
            emb_v_out, emb_g_out, emb_l_out, v_seq_len, g_seq_len, l_seq_len, n_head_self_attn_mask, _checkpoints = \
                self._gen_input(image_input=image_input, image_mask=image_mask,
                                text_input=text_input, text_mask=text_mask, gather_idx=gather_idx)
            all_checkpoints.extend(_checkpoints)

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

            vgl_seq_len = v_seq_len + g_seq_len + l_seq_len
            enc_v_out = paddle.slice(
                input=enc_vgl_out, axes=[1], starts=[0], ends=[v_seq_len])
            enc_g_out = paddle.slice(
                input=enc_vgl_out, axes=[1], starts=[v_seq_len], ends=[v_seq_len + g_seq_len])
            enc_l_out = paddle.slice(
                input=enc_vgl_out, axes=[1], starts=[v_seq_len + g_seq_len], ends=[v_seq_len + g_seq_len + l_seq_len])
            return enc_v_out, enc_g_out, enc_l_out, all_checkpoints, vgl_seq_len
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

    def text_encode(self, text_input, text_mask, gather_idx=None):
        assert text_mask is not None, "text_mask should not be none"
        text_self_attn_mask = text_mask
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

        return quantized, encoding_indices_reshaped

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

        return quantized, encoding_indices_reshaped

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
                self.text_encode(text_input, text_mask, gather_idx=gather_idx)
            _checkpoints.extend(text_checkpoints)

        if image_input is not None and text_input is not None:  # for img2txt training
            if self._grounding_method == "normal":
                self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.vector_quantizer(patch_embeddings)
            elif self._grounding_method == "topk":
                self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.topk_vector_quantizer(patch_embeddings, self._K)
            elif self._grounding_method == "optimal":
                self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.vector_quantizer(patch_embeddings)
            else:
                raise ValueError("%s is not supported!!!" % self._grounding_method)

            batch_size = paddle.shape(self.grounded_enc_out)[0]
            g_seq_len = paddle.shape(self.grounded_enc_out)[1]

            # (batch_size, v_seq_len, g_seq_len)
            img_grounded_attn_mask = paddle.ones(shape=[batch_size, vit_seq_len, g_seq_len], dtype='float32')
            # (batch_size, v_seq_len, l_seq_len)
            img_text_attn_mask = paddle.zeros(shape=[batch_size, vit_seq_len, text_seq_len], dtype='float32')
            # (batch_size, g_seq_len, v_seq_len)
            grounded_img_attn_mask = paddle.ones(shape=[batch_size, g_seq_len, vit_seq_len], dtype='float32')
            # (batch_size, g_seq_len, g_seq_len)
            grounded_self_attn_mask = paddle.ones(shape=[batch_size, g_seq_len, g_seq_len], dtype='float32')
            # (batch_size, g_seq_len, l_seq_len)
            grounded_text_attn_mask = paddle.zeros(shape=[batch_size, g_seq_len, text_seq_len], dtype='float32')
            # (batch_size, l_seq_len, v_seq_len)
            text_img_attn_mask = paddle.ones(shape=[batch_size, text_seq_len, vit_seq_len], dtype='float32')
            # (batch_size, l_seq_len, g_seq_len)
            text_grounded_attn_mask = paddle.ones(shape=[batch_size, text_seq_len, g_seq_len], dtype='float32')

            image_row = paddle.concat([image_self_attn_mask, img_grounded_attn_mask, img_text_attn_mask], axis=2)
            grounded_row = paddle.concat([grounded_img_attn_mask, grounded_self_attn_mask, grounded_text_attn_mask],
                                         axis=2)
            text_row = paddle.concat([text_img_attn_mask, text_grounded_attn_mask, text_mask], axis=2)
            vgl_self_attn_mask = paddle.concat([image_row, grounded_row, text_row], axis=1)
            vgl_self_attn_mask = paddle.scale(
                x=vgl_self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
            n_head_self_attn_mask = paddle.stack(
                x=[vgl_self_attn_mask] * self._n_head, axis=1)
            n_head_self_attn_mask.stop_gradient = True

            return self.vit_enc_out, self.grounded_enc_out, self.text_enc_out, \
                   vit_seq_len, g_seq_len, text_seq_len, n_head_self_attn_mask, _checkpoints

        elif image_input is not None and text_input is None:  # for img2txt inference
            if self._grounding_method == "normal":
                self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.vector_quantizer(patch_embeddings)
            elif self._grounding_method == "topk":
                self.grounded_enc_out, self.grounded_encoding_indices = \
                    self.topk_vector_quantizer(patch_embeddings, self._K)
            elif self._grounding_method == "optimal":
                self.grounded_enc_out, self.grounded_encoding_indices = \
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

        elif image_input is None and text_input is not None:  # for img2txt step-by-step decoding
            assert decoding_step is True, "decoding_step must be True"
            assert grounded_decoding_mask is not None, "grounded_decoding_mask should not be none"
            self_attn_mask = paddle.scale(
                x=grounded_decoding_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
            n_head_self_attn_mask = paddle.stack(
                x=[self_attn_mask] * self._n_head, axis=1)
            n_head_self_attn_mask.stop_gradient = True

            return self.text_enc_out, text_seq_len, n_head_self_attn_mask, _checkpoints

    def get_checkpoints(self):
        return self.all_checkpoints

    def get_text_sequence_output(self):
        return self._enc_l_out


class BaselineForImg2Txt(object):
    def __init__(self,
                 image_input=None,
                 image_mask=None,
                 text_input=None,
                 text_mask=None,
                 config=None,
                 weight_sharing=True,
                 decoding=False,
                 gather_idx=None,
                 grounded_encoder_trainable=True,
                 vit_encoder_trainable=True,
                 text_encoder_trainable=True,
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

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = paddle.fluid.initializer.TruncatedNormalInitializer(scale=config['initializer_range'])
        self._bias_initializer = paddle.fluid.initializer.ConstantInitializer(value=0.0)

        assert text_input is not None or image_input is not None, "text_input and image_input cannot both be None"

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
        if text_input is None and image_input is not None:  # for img2txt inference
            self._enc_v_out, self.all_checkpoints, self.out_seq_len = self.encode(image_input=image_input,
                                                                                  image_mask=image_mask,
                                                                                  gather_idx=gather_idx)
        elif text_input is not None and image_input is None:  # for img2txt step-by-step decoding
            self._enc_l_out, self.all_checkpoints = self.encode(text_input=text_input,
                                                                text_mask=text_mask,
                                                                gather_idx=gather_idx)
        else:  # for img2txt training
            self._enc_v_out, self._enc_l_out, self.all_checkpoints, self.out_seq_len = \
                self.encode(text_input=text_input,
                            text_mask=text_mask,
                            gather_idx=gather_idx,
                            image_input=image_input,
                            image_mask=image_mask)

    def encode(self, text_input=None, text_mask=None, gather_idx=None,
               image_input=None, image_mask=None, decoding_step=False, grounded_decoding_mask=None):
        all_checkpoints = []
        # padding id in vocabulary must be set to 0
        if text_input is None and image_input is not None:  # for img2txt inference
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
            return enc_v_out, all_checkpoints, v_seq_len

        elif image_input is None and text_input is not None:  # for img2txt step-by-step decoding
            assert decoding_step is True, 'decoding_step must be True'
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

        elif image_input is not None and text_input is not None:  # for img2txt training
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

            vl_seq_len = v_seq_len + l_seq_len
            enc_v_out = paddle.slice(
                input=enc_vl_out, axes=[1], starts=[0], ends=[v_seq_len])
            enc_l_out = paddle.slice(
                input=enc_vl_out, axes=[1], starts=[v_seq_len], ends=[v_seq_len + l_seq_len])
            return enc_v_out, enc_l_out, all_checkpoints, vl_seq_len
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

    def text_encode(self, text_input, text_mask, gather_idx=None):
        assert text_mask is not None, "text_mask should not be none"
        text_self_attn_mask = text_mask
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
            self.vit_enc_out, vit_seq_len, nhead_image_self_attn_mask, vit_checkpoints = self.vit_encode(image_input,
                                                                                                         image_mask)
            _checkpoints.extend(vit_checkpoints)

        if text_input is not None:
            # textual part
            self.text_enc_out, text_seq_len, _, text_checkpoints = self.text_encode(text_input,
                                                                                    text_mask,
                                                                                    gather_idx=gather_idx)
            _checkpoints.extend(text_checkpoints)

        if image_input is not None and text_input is not None:
            batch_size = paddle.shape(self.text_enc_out)[0]
            # (batch_size, v_seq_len, v_seq_len)
            image_self_attn_mask = paddle.matmul(x=paddle.transpose(image_mask, perm=[0, 2, 1]), y=image_mask)
            # (batch_size, v_seq_len, l_seq_len)
            img_text_attn_mask = paddle.zeros(shape=[batch_size, vit_seq_len, text_seq_len], dtype='float32')
            # (batch_size, l_seq_len, v_seq_len)
            text_img_attn_mask = paddle.ones(shape=[batch_size, text_seq_len, vit_seq_len], dtype='float32')

            image_row = paddle.concat([image_self_attn_mask, img_text_attn_mask], axis=2)
            text_row = paddle.concat([text_img_attn_mask, text_mask], axis=2)
            vl_self_attn_mask = paddle.concat([image_row, text_row], axis=1)
            vl_self_attn_mask = paddle.scale(
                x=vl_self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
            n_head_self_attn_mask = paddle.stack(
                x=[vl_self_attn_mask] * self._n_head, axis=1)
            n_head_self_attn_mask.stop_gradient = True

            return self.vit_enc_out, self.text_enc_out, vit_seq_len, text_seq_len, n_head_self_attn_mask, _checkpoints

        elif image_input is not None and text_input is None:
            return self.vit_enc_out, vit_seq_len, nhead_image_self_attn_mask, _checkpoints

        elif image_input is None and text_input is not None:
            assert decoding_step is True, "decoding_step must be True"
            assert grounded_decoding_mask is not None, "grounded_decoding_mask should not be none"
            self_attn_mask = paddle.scale(
                x=grounded_decoding_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
            n_head_self_attn_mask = paddle.stack(
                x=[self_attn_mask] * self._n_head, axis=1)
            n_head_self_attn_mask.stop_gradient = True

            return self.text_enc_out, text_seq_len, n_head_self_attn_mask, _checkpoints

    def get_checkpoints(self):
        return self.all_checkpoints

    def get_text_sequence_output(self):
        return self._enc_l_out
