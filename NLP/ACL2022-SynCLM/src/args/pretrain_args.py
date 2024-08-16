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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from utils.args import ArgumentGroup

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, None,
                "Init pre-training params which preforms fine-tuning from. If the "
                "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints", str, "checkpoints", "Path to save checkpoints.")
model_g.add_arg("weight_sharing", bool, True, "If set, share weights between word embedding and masked lm.")
model_g.add_arg("roberta_vocab_file", str, './model_files/dict/roberta_base_en.vocab.txt', "roberta vocab")
model_g.add_arg("encoder_json_file", str, './model_files/dict/roberta_base_en.encoder.json', 'bpt map')
model_g.add_arg("vocab_bpe_file", str, './model_files/dict/roberta_base_en.vocab.bpe', "vocab bpe")
model_g.add_arg("synclm_config_path", str, "./model_files/config/roberta_base_en.json",
                "The file to save roberta configuration.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 100, "Number of epoches for training.")
train_g.add_arg("learning_rate", float, 0.0001, "Learning rate used to train with warmup.")
train_g.add_arg("lr_scheduler", str, "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
train_g.add_arg("num_train_steps", int, 1000000, "Total steps to perform pretraining.")
train_g.add_arg("warmup_steps", int, 5000, "Total steps to perform warmup when pretraining.")
train_g.add_arg("save_steps", int, 10000, "The steps interval to save checkpoints.")
train_g.add_arg("validation_steps", int, 1000, "The steps interval to evaluate model performance.")
train_g.add_arg("use_fuse", bool, False, "Whether to use fuse_allreduce_ops.")
train_g.add_arg("nccl_comm_num", int, 1, "NCCL comm num.")
train_g.add_arg("hierarchical_allreduce_inter_nranks", int, 8, "Hierarchical allreduce inter ranks.")
train_g.add_arg("use_hierarchical_allreduce", bool, False, "Use hierarchical allreduce or not.")
train_g.add_arg("use_fp16", bool, False, "Whether to use fp16 mixed precision training.")
train_g.add_arg("use_dynamic_loss_scaling", bool, False, "Whether to use dynamic loss scaling.")
train_g.add_arg("init_loss_scaling", float, 1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
train_g.add_arg("init_step", int, 0,
                "init step for continue train model, only valid when load checkpoint")
train_g.add_arg("incr_every_n_steps", int, 1000, "Increases loss scaling every n consecutive.")
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

# args for phrase and tree loss
train_g.add_arg("att_layer", str, "-1", "attention layer for phrase sample")
train_g.add_arg("phrase_max_neg_num", int, 10, "the max num of negative samples for phrase sample")
train_g.add_arg("tree_max_sub_num", int, 10, "the max num of sub nodes for tree sample")
train_g.add_arg("tree_max_neg_num", int, 10, "the max num of negative samples for tree sample")


log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 10, "The steps interval to print loss.")
log_g.add_arg("num_iteration_per_drop_scope", int, 10, "The steps to clear temporary variable.")
log_g.add_arg("verbose", bool, False, "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_filelist", str, "", "Path to training filelist.")
data_g.add_arg("valid_filelist", str, "", "Path to valid filelist.")
data_g.add_arg("test_filelist", str, "", "Path to test filelist.")
data_g.add_arg("max_seq_len", int, 512, "Number of words of the longest sequence.")
data_g.add_arg("batch_size", int, 16, "Total examples' number in batch for training. see also --in_tokens.")
data_g.add_arg("in_tokens", bool, False,
               "If set, the batch size will be the maximum number of tokens in one batch. "
               "Otherwise, it will be the maximum number of examples in one batch.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("visualdl_log", bool, False, "If set, use visualdl_log on paddlecloud.")
run_type_g.add_arg("is_distributed", bool, False, "If set, then start distributed training.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor", bool, False, "If set, use fast parallel executor (in experiment).")
run_type_g.add_arg("do_test", bool, False, "Whether to perform evaluation on test data set.")
run_type_g.add_arg("random_seed", int, 0, "Random seed.")

# yapf: enable
