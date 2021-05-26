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
"""Training the GRAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import shutil
import argparse
import multiprocessing
import os
import time
import logging
import numpy as np
import paddle.fluid as fluid

from reader.vocab_reader import Vocabulary
from reader.data_reader import DataReader
from model.gran_model import GRANModel
from optimization import optimization
from evaluation import generate_ground_truth, batch_evaluation, compute_metrics
from utils.args import ArgumentGroup, print_arguments
from utils.init import init_pretraining_params, init_checkpoint

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

# yapf: disable
parser = argparse.ArgumentParser()
model_g = ArgumentGroup(parser, "model", "model and checkpoint configuration.")
model_g.add_arg("num_hidden_layers",       int,    12,        "Number of hidden layers.")
model_g.add_arg("num_attention_heads",     int,    4,         "Number of attention heads.")
model_g.add_arg("hidden_size",             int,    256,       "Hidden size.")
model_g.add_arg("intermediate_size",       int,    512,       "Intermediate size.")
model_g.add_arg("hidden_act",              str,    "gelu",    "Hidden act.")
model_g.add_arg("hidden_dropout_prob",     float,  0.1,       "Hidden dropout ratio.")
model_g.add_arg("attention_dropout_prob",  float,  0.1,       "Attention dropout ratio.")
model_g.add_arg("initializer_range",       float,  0.02,      "Initializer range.")
model_g.add_arg("vocab_size",              int,    None,      "Size of vocabulary.")
model_g.add_arg("num_relations",           int,    None,      "Number of relations.")
model_g.add_arg("num_edges",               int,    5,
                "Number of edge types, typically fixed to 5: no edge (0), relation-subject (1),"
                "relation-object (2), relation-attribute (3), attribute-value (4).")
model_g.add_arg("max_seq_len",             int,    None,      "Max sequence length.")
model_g.add_arg("max_arity",               int,    None,      "Max arity.")
model_g.add_arg("entity_soft_label",       float,  1.0,       "Label smoothing rate for masked entities.")
model_g.add_arg("relation_soft_label",     float,  1.0,       "Label smoothing rate for masked relations.")
model_g.add_arg("weight_sharing",          bool,   True,      "If set, share masked lm weights with node embeddings.")
model_g.add_arg("init_checkpoint",         str,    None,      "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str,    None,
                "Init pre-training params which preforms fine-tuning from. "
                "If 'init_checkpoint' has been set, this argument wouldn't be valid.")
model_g.add_arg("checkpoints",             str,    "ckpts",   "Path to save checkpoints.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("batch_size",        int,    1024,                   "Batch size.")
train_g.add_arg("epoch",             int,    100,                    "Number of training epochs.")
train_g.add_arg("learning_rate",     float,  5e-4,                   "Learning rate with warmup.")
train_g.add_arg("lr_scheduler",      str,    "linear_warmup_decay",  "scheduler of learning rate.",
                choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("warmup_proportion", float,  0.1,                    "Proportion of training steps for lr warmup.")
train_g.add_arg("weight_decay",      float,  0.01,                   "Weight decay rate for L2 regularizer.")
train_g.add_arg("use_fp16",          bool,   False,                  "Whether to use fp16 mixed precision training.")
train_g.add_arg("loss_scaling",      float,  1.0,
                "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps",          int,    1000,    "Step intervals to print loss.")
log_g.add_arg("verbose",             bool,   False,   "Whether to output verbose log.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options.")
data_g.add_arg("train_file",              str,    None,  "Data for training.")
data_g.add_arg("predict_file",            str,    None,  "Data for prediction.")
data_g.add_arg("ground_truth_path",       str,    None,  "Path to ground truth.")
data_g.add_arg("vocab_path",              str,    None,  "Path to vocabulary.")


run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
run_type_g.add_arg("use_fast_executor",            bool,   False, "If set, use fast parallel executor.")
run_type_g.add_arg("num_iteration_per_drop_scope", int,    1,     "Iteration intervals to clean up temporary variables.")
run_type_g.add_arg("do_train",                     bool,   False, "Whether to perform training.")
run_type_g.add_arg("do_predict",                   bool,   False, "Whether to perform prediction.")

args = parser.parse_args()
# yapf: enable.


def make_tarfile(folder_path, remove=True):
    target_path = folder_path + ".tar"
    with tarfile.open(target_path, "w:") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    if remove:
        shutil.rmtree(folder_path, ignore_errors=True)


def create_model(pyreader_name, config):
    pyreader = fluid.layers.py_reader(
        capacity=60,
        shapes=[[-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, 1], [-1, 1], [-1, 1],
                [args.max_seq_len, args.max_seq_len, 1]],
        dtypes=[
            'int64', 'float32', 'int64', 'int64', 'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)
    (input_ids, input_mask, mask_position, mask_label, mask_type, edge_labels) \
        = fluid.layers.read_file(pyreader)

    gran = GRANModel(
        input_ids=input_ids,
        input_mask=input_mask,
        edge_labels=edge_labels,
        config=config,
        weight_sharing=args.weight_sharing,
        use_fp16=args.use_fp16)

    loss, fc_out = gran.get_mask_lm_output(
        mask_pos=mask_position, mask_label=mask_label, mask_type=mask_type)
    if args.use_fp16 and args.loss_scaling > 1.0:
        loss = loss * args.loss_scaling

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=mask_label, dtype='int64', shape=[1], value=1)
    num_seqs = fluid.layers.reduce_sum(input=batch_ones)

    return pyreader, loss, fc_out, num_seqs


def predict(test_exe, test_program, test_pyreader, fetch_list, all_features, vocabulary):
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    eval_result_file = os.path.join(args.checkpoints, "eval_result.json")

    gt_dict = generate_ground_truth(
        ground_truth_path=args.ground_truth_path,
        vocabulary=vocabulary,
        max_arity=args.max_arity,
        max_seq_length=args.max_seq_len)

    step = 0
    global_idx = 0
    ent_lst = []
    rel_lst = []
    _2_r_lst = []
    _2_ht_lst = []
    _n_r_lst = []
    _n_ht_lst = []
    _n_a_lst = []
    _n_v_lst = []
    test_pyreader.start()
    while True:
        try:
            batch_results = []
            np_fc_out = test_exe.run(fetch_list=fetch_list, program=test_program)[0]
            for idx in range(np_fc_out.shape[0]):
                logits = [float(x) for x in np_fc_out[idx].flat]
                batch_results.append(logits)
            ent_ranks, rel_ranks, _2_r_ranks, _2_ht_ranks, \
            _n_r_ranks, _n_ht_ranks, _n_a_ranks, _n_v_ranks = batch_evaluation(
                global_idx, batch_results, all_features, gt_dict)
            ent_lst.extend(ent_ranks)
            rel_lst.extend(rel_ranks)
            _2_r_lst.extend(_2_r_ranks)
            _2_ht_lst.extend(_2_ht_ranks)
            _n_r_lst.extend(_n_r_ranks)
            _n_ht_lst.extend(_n_ht_ranks)
            _n_a_lst.extend(_n_a_ranks)
            _n_v_lst.extend(_n_v_ranks)
            if step % 10 == 0:
                logger.info("Processing prediction steps: %d examples: %d" % (step, global_idx))
            step += 1
            global_idx += np_fc_out.shape[0]
        except fluid.core.EOFException:
            test_pyreader.reset()
            break

    eval_result = compute_metrics(
        ent_lst=ent_lst,
        rel_lst=rel_lst,
        _2_r_lst=_2_r_lst,
        _2_ht_lst=_2_ht_lst,
        _n_r_lst=_n_r_lst,
        _n_ht_lst=_n_ht_lst,
        _n_a_lst=_n_a_lst,
        _n_v_lst=_n_v_lst,
        eval_result_file=eval_result_file
    )

    return eval_result


def main(args):
    if not (args.do_train or args.do_predict):
        raise ValueError("For args `do_train` and `do_predict`, at "
                         "least one of them must be True.")
    config = vars(args)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    vocabulary = Vocabulary(
        vocab_file=args.vocab_path,
        num_relations=args.num_relations,
        num_entities=args.vocab_size - args.num_relations - 2)

    # Init program
    startup_prog = fluid.Program()

    if args.do_train:
        train_data_reader = DataReader(
            data_path=args.train_file,
            max_arity=args.max_arity,
            max_seq_length=args.max_seq_len,
            batch_size=args.batch_size,
            is_training=True,
            shuffle=True,
            dev_count=dev_count,
            epoch=args.epoch)

        num_train_instances = train_data_reader.total_instance
        max_train_steps = args.epoch * num_train_instances // args.batch_size // dev_count
        warmup_steps = int(max_train_steps * args.warmup_proportion)
        logger.info("Device count: %d" % dev_count)
        logger.info("Num train instances: %d" % num_train_instances)
        logger.info("Max train steps: %d" % max_train_steps)
        logger.info("Num warmup steps: %d" % warmup_steps)

        # Create model and set optimization for training
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, loss, _, num_seqs = create_model(
                    pyreader_name='train_reader', config=config)
                scheduled_lr = optimization(
                    loss=loss,
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    loss_scaling=args.loss_scaling)
                fluid.memory_optimize(train_program, skip_opt_set=[loss.name, num_seqs.name])

        if args.verbose:
            lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            logger.info("Theoretical memory usage in training:  %.3f - %.3f %s" %
                        (lower_mem, upper_mem, unit))

    if args.do_predict:
        # Create model for prediction
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, _, fc_out, num_seqs = create_model(
                    pyreader_name='test_reader', config=config)
                fluid.memory_optimize(test_prog, skip_opt_set=[fc_out.name, num_seqs.name])
        test_prog = test_prog.clone(for_test=True)

    exe.run(startup_prog)

    # Init checkpoints
    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            logger.info(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
    elif args.do_predict:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing prediction!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog,
            use_fp16=args.use_fp16)

    # Run training
    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = args.use_fast_executor
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=loss.name,
            exec_strategy=exec_strategy,
            main_program=train_program)

        train_pyreader.decorate_tensor_provider(
            train_data_reader.data_generator(vocabulary=vocabulary))

        train_pyreader.start()
        steps = 0
        total_cost, total_num_seqs = [], []
        time_begin = time.time()
        while steps < max_train_steps:
            try:
                steps += 1
                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        fetch_list = [loss.name, num_seqs.name]
                    else:
                        fetch_list = [
                            loss.name, scheduled_lr.name, num_seqs.name
                        ]
                else:
                    fetch_list = []

                outputs = train_exe.run(fetch_list=fetch_list)

                if steps % args.skip_steps == 0:
                    if warmup_steps <= 0:
                        np_loss, np_num_seqs = outputs
                    else:
                        np_loss, np_lr, np_num_seqs = outputs
                    total_cost.extend(np_loss * np_num_seqs)
                    total_num_seqs.extend(np_num_seqs)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size(
                        )
                        verbose += "learning rate: %f" % (
                            np_lr[0]
                            if warmup_steps > 0 else args.learning_rate)
                        logger.info(verbose)

                    time_end = time.time()
                    used_time = time_end - time_begin
                    current_instance, epoch = train_data_reader.get_progress()

                    logger.info("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                                "speed: %f steps/s" %
                                (epoch, current_instance, num_train_instances, steps,
                                 np.sum(total_cost) / np.sum(total_num_seqs),
                                 args.skip_steps / used_time))
                    total_cost, total_num_seqs = [], []
                    time_begin = time.time()

                if steps == max_train_steps:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)
                    make_tarfile(save_path)
            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints,
                                         "step_" + str(steps) + "_final")
                fluid.io.save_persistables(exe, save_path, train_program)
                make_tarfile(save_path)
                train_pyreader.reset()
                break

    # Run prediction
    if args.do_predict:
        test_data_reader = DataReader(
            data_path=args.predict_file,
            max_arity=args.max_arity,
            max_seq_length=args.max_seq_len,
            batch_size=args.batch_size,
            is_training=False,
            shuffle=False,
            dev_count=1,
            epoch=1)

        test_pyreader.decorate_tensor_provider(
            test_data_reader.data_generator(vocabulary=vocabulary))

        eval_performance = predict(
            test_exe=exe,
            test_program=test_prog,
            test_pyreader=test_pyreader,
            fetch_list=[fc_out.name],
            all_features=test_data_reader.get_features(
                vocabulary=vocabulary),
            vocabulary=vocabulary)

        all_entity = "ENTITY\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
            eval_performance['entity']['mrr'],
            eval_performance['entity']['hits1'],
            eval_performance['entity']['hits3'],
            eval_performance['entity']['hits5'],
            eval_performance['entity']['hits10'])

        all_relation = "RELATION\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
            eval_performance['relation']['mrr'],
            eval_performance['relation']['hits1'],
            eval_performance['relation']['hits3'],
            eval_performance['relation']['hits5'],
            eval_performance['relation']['hits10'])

        all_ht = "HEAD/TAIL\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
            eval_performance['ht']['mrr'],
            eval_performance['ht']['hits1'],
            eval_performance['ht']['hits3'],
            eval_performance['ht']['hits5'],
            eval_performance['ht']['hits10'])

        all_r = "PRIMARY_R\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
            eval_performance['r']['mrr'],
            eval_performance['r']['hits1'],
            eval_performance['r']['hits3'],
            eval_performance['r']['hits5'],
            eval_performance['r']['hits10'])

        logger.info("\n-------- Evaluation Performance --------\n%s\n%s\n%s\n%s\n%s" % (
            "\t".join(["TASK", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
            all_ht, all_r, all_entity, all_relation))


if __name__ == '__main__':
    print_arguments(args)
    main(args)
