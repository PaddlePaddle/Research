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
File: retrieval_grounded_args.py
Author: liwei(liwei85@baidu.com)
Date: 2021-08-23 14:28
Desc: args for retrieval task
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from utils.args import ArgumentGroup

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("run_random", bool, False, "run model with random params")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")
model_g.add_arg("save_checkpoints", bool, True, "Whether to save checkpoints")
model_g.add_arg("weight_sharing", bool, True, "If set, share weights between word embedding and masked lm.")
model_g.add_arg("unimo_vocab_file", str, './model_files/dict/unimo_en.vocab.txt', "unimo vocab")
model_g.add_arg("encoder_json_file", str, './model_files/dict/unimo_en.encoder.json', 'bpt map')
model_g.add_arg("vocab_bpe_file", str, './model_files/dict/unimo_en.vocab.bpe', "vocab bpe")
model_g.add_arg("unimo_config_path", str, "./model_files/config/unimo_base_en.json",
                "The file to save unimo configuration.")
train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train.")
train_g.add_arg("learning_rate_scale", float, 0.1, "Learning rate decay scale.")
train_g.add_arg("lr_scheduler", str, "scale_by_epoch_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay', 'scale_by_epoch_decay'])
train_g.add_arg("learning_rate_decay_epoch1", int, 24, "Learning rate decay epoch1.")
train_g.add_arg("learning_rate_decay_epoch2", int, 32, "Learning rate decay epoch2.")
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_step", int, 1, "warmup_step, 1 for scale_by_epoch_decay, 0 for others")

train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
train_g.add_arg("nccl_comm_num", int, 1, "NCCL comm num.")
train_g.add_arg("hierarchical_allreduce_inter_nranks", int, 8, "Hierarchical allreduce inter ranks.")
train_g.add_arg("use_hierarchical_allreduce", bool, False, "Use hierarchical allreduce or not.")
train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
train_g.add_arg("use_dynamic_loss_scaling", bool, False, "Whether to use dynamic loss scaling.")
train_g.add_arg("use_sigmoid", bool, True, "Whether to use sigmoid before loss")
train_g.add_arg("init_loss_scaling", float, 1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
train_g.add_arg("incr_every_n_steps", int, 100, "Increases loss scaling every n consecutive.")
train_g.add_arg("decr_every_n_nan_or_inf", int, 2,
                "Decreases loss scaling every n accumulated steps with nan or inf gradients.")
train_g.add_arg("incr_ratio", float, 2.0,
                "The multiplier to use when increasing the loss scaling.")
train_g.add_arg("decr_ratio", float, 0.8,
                "The less-than-one-multiplier to use when decreasing.")

# args for adam optimizer
train_g.add_arg("beta1", float, 0.9, "beta1 for adam")
train_g.add_arg("beta2", float, 0.98, "beta2 for adam.")
train_g.add_arg("epsilon", float, 1e-06, "epsilon for adam.")
train_g.add_arg("use_fuse", bool, False, "Whether to use fuse_allreduce_ops.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")
log_g.add_arg("eval_dir", str, "", "eval_dir to save tmp data")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("samples_num", int, 20, "neg sample num.")
data_g.add_arg("train_image_caption", str, None, "Path to training data.")
data_g.add_arg("test_image_caption", str, None, "Path to test data.")
data_g.add_arg("dev_image_caption", str, None, "Path to validation data.")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest seqence.")
data_g.add_arg("batch_size", int, 32, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("test_batch_size", int, 24, "Total examples' number in batch for testing.")
data_g.add_arg("do_lower_case", bool, True,
               "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
data_g.add_arg("random_seed", int, 0, "Random seed.")
data_g.add_arg("max_img_len", int, 37, "Image feature size==2048.")
data_g.add_arg("scale_circle", float, "1.0", "The scale factor in circle loss function, only use in circle loss mode")
data_g.add_arg("margin", float, "0.2", "The margin value in loss function")
data_g.add_arg("max_neg_cap_num", int, 0, "max_neg_cap_num")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("is_distributed", bool, False, "If set, then start distributed training.")
run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("num_iteration_per_drop_scope", int, 10, "Iteration intervals to drop scope.")
run_type_g.add_arg("do_train", bool, True, "Whether to perform training.")
run_type_g.add_arg("do_val", bool, True, "Whether to perform evaluation on dev data set.")
run_type_g.add_arg("do_test", bool, True, "Whether to perform evaluation on test data set.")
run_type_g.add_arg("use_multi_gpu_test", bool, False, "Whether to perform evaluation using multiple gpu cards")
run_type_g.add_arg("eval_mertrics", str, "recall@k", "eval_mertrics")
run_type_g.add_arg("zero_shot", bool, True, "Whether to do zero shot.")

# new add args
image_g = ArgumentGroup(parser, "image", "image configuration options")
image_g.add_arg("resolution", int, 16, "used for VIT model")
image_g.add_arg("image_size", int, 224, "image size")
image_g.add_arg("num_codebook", int, 2048, "size of image codebook")
image_g.add_arg("grounding_method", str, "topk",
                "type of grounding method", choices=['normal', 'topk', 'optimal'])
image_g.add_arg("model_type", str, "grounded",
                "type of base model", choices=['grounded', 'baseline'])
image_g.add_arg("topk_value", int, 100, "pamater for topk grounding")
image_g.add_arg("with_cmcl", bool, True, "Whether to use CMCL")
image_g.add_arg("with_cmcl_projection", bool, True, "Whether to use linear projection for cmcl")
image_g.add_arg("with_grounding_projection", bool, True, "Whether to use linear projection before cmcl")
image_g.add_arg("cmcl_share_parameters", bool, True, "Whether to share linear projection for cmcl and img-text matching")
image_g.add_arg("cmcl_score_weight", float, "0.1", "The weight for cmcl score")
image_g.add_arg("with_grounding_pos", bool, False, "Whether to use pos_emb for grounding tokens")
model_g.add_arg("text_enc_layers", str, '0,1,2,3,4,5', "text encoder layers")
model_g.add_arg("grounding_enc_layers", str, '6,7,8,9,10,11', "grounding encoder layers")
image_g.add_arg("use_recompute", bool, False, "Whether to use use_recompute")
# yapf: enable
