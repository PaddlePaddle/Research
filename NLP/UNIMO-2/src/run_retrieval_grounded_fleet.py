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
File: run_retrieval_grounded_fleet.py
Author: liwei(liwei85@baidu.com)
Date: 2021-10-22 14:26
Desc: finetuning for image / text retrieval
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing

os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.98"
os.environ['FLAGS_sync_nccl_allreduce'] = "1"
os.environ['FLAGS_eager_delete_tensor_gb'] = "0"
os.environ['FLAGS_fuse_parameter_memory_size'] = "32"
os.environ['FLAGS_fuse_parameter_groups_size'] = "50"
os.environ['FLAGS_check_nan_inf'] = "0"

import paddle

paddle.enable_static()
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker
import numpy as np
import random

from reader.retrieval_grounded_reader import RetrievalTrainReader, RetrievalTestReader
from model.unimo_grounded import VlConfig
from model.roberta_tokenization import GptBpeTokenizer
from finetune.retrieval_grounded_fleet import create_model, evaluate
from utils.optimization_fleet import optimization
from utils.args import print_arguments
from utils.utils import get_time
from utils.init import init_pretraining_params, init_checkpoint, check_pretraining_params
from args.retrieval_grounded_args import parser

args = parser.parse_args()


def main(args):
    """main"""
    model_config = VlConfig(args.unimo_config_path)
    model_config['image_size'] = args.image_size
    model_config["resolution"] = args.resolution
    model_config['num_codebook'] = args.num_codebook
    model_config.print_config()

    print("FLAGS_selected_gpus:", os.environ.get('FLAGS_selected_gpus'))
    place = paddle.CUDAPlace(int(os.environ.get('FLAGS_selected_gpus', 0)))

    dist_strategy = paddle.distributed.fleet.DistributedStrategy()
    # dist_strategy.without_graph_optimization = True
    dist_strategy.fuse_all_reduce_ops = True

    exec_strategy = paddle.static.ExecutionStrategy()
    exec_strategy.num_threads = 4 if args.use_fp16 else 2  # 2 for fp32 4 for fp16
    exec_strategy.num_iteration_per_drop_scope = min(args.num_iteration_per_drop_scope, args.skip_steps)
    # exec_strategy.num_iteration_per_run = 10
    dist_strategy.execution_strategy = exec_strategy

    build_strategy = paddle.static.BuildStrategy()
    dist_strategy.build_strategy = build_strategy

    if args.use_fp16:
        dist_strategy.amp = True
        dist_strategy.amp_configs = {
            "init_loss_scaling": args.init_loss_scaling,
            "use_dynamic_loss_scaling": args.use_dynamic_loss_scaling,
            "custom_black_list": ['layer_norm', 'arg_max', 'argmax'],
            "custom_black_varnames": ["loss"]
        }

    if args.use_recompute:
        print("using recompute.")
        dist_strategy.recompute = True
        build_strategy.enable_sequential_execution = True

    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role_maker=role, strategy=dist_strategy)

    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()
    print("worker_num %d" % worker_num)
    print("worker_index %d" % worker_index)

    tokenizer = GptBpeTokenizer(vocab_file=args.unimo_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=args.do_lower_case)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val`, `do_test`, at "
                         "least one of them must be True.")

    startup_prog = paddle.static.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        paddle.seed(args.random_seed)

    if args.do_train:
        train_data_reader = RetrievalTrainReader(tokenizer=tokenizer,
                                                 args=args,
                                                 image_caption=args.train_image_caption,
                                                 image_size=args.image_size,
                                                 resolution=args.resolution)
        train_data_generator = train_data_reader.data_generator()
        num_train_examples, captions_num, image_num = train_data_reader.get_num_examples()
        step_num_per_epoch = num_train_examples // args.batch_size // worker_num
        max_train_steps = args.epoch * step_num_per_epoch
        args.learning_rate_decay_step1 = args.learning_rate_decay_epoch1 * step_num_per_epoch
        args.learning_rate_decay_step2 = args.learning_rate_decay_epoch2 * step_num_per_epoch

        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)

        train_program = paddle.static.Program()
        lr_boundaries = [args.learning_rate_decay_step1, args.learning_rate_decay_step2]
        lr_value = [args.learning_rate * args.learning_rate_scale ** i for i in range(len(lr_boundaries) + 1)]

        with paddle.static.program_guard(train_program, startup_prog):
            train_pyreader, graph_vars, checkpoints = create_model(
                args,
                phase='train',
                vl_config=model_config,
                samples_num=args.samples_num + 1)

            if args.use_recompute:
                dist_strategy.recompute_configs = {"checkpoints": checkpoints}

            scheduled_lr = optimization(
                loss=graph_vars["loss"],
                warmup_steps=args.warmup_step,
                num_train_steps=max_train_steps,
                learning_rate=args.learning_rate,
                train_program=train_program,
                weight_decay=args.weight_decay,
                scheduler=args.lr_scheduler,
                beta1=args.beta1,
                beta2=args.beta2,
                epsilon=args.epsilon,
                dist_strategy=dist_strategy,
                boundaries=lr_boundaries,
                values=lr_value)

    if args.do_val or args.do_test:
        test_prog = paddle.static.Program()
        with paddle.static.program_guard(test_prog, startup_prog):
            test_pyreader, test_graph_vars, _ = create_model(
                args,
                phase='dev',
                vl_config=model_config,
                samples_num=1)
        test_prog = test_prog.clone(for_test=True)

        if args.do_val:
            dev_data_reader = RetrievalTestReader(tokenizer=tokenizer,
                                                  args=args,
                                                  image_caption=args.dev_image_caption,
                                                  image_size=args.image_size,
                                                  resolution=args.resolution)
            dev_data_generator = dev_data_reader.data_generator()
        if args.do_test:
            test_data_reader = RetrievalTestReader(tokenizer=tokenizer,
                                                   args=args,
                                                   image_caption=args.dev_image_caption,
                                                   image_size=args.image_size,
                                                   resolution=args.resolution)
            test_data_generator = test_data_reader.data_generator()

    exe = paddle.static.Executor(place)
    exe.run(startup_prog)

    if args.do_train:
        if not args.run_random:
            if args.init_checkpoint and args.init_pretraining_params:
                print(
                    "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                    "both are set! Only arg 'init_checkpoint' is made valid.")
            if args.init_checkpoint:
                check_pretraining_params(args.init_checkpoint, train_program)
                init_checkpoint(
                    exe,
                    args.init_checkpoint,
                    main_program=train_program)
            elif args.init_pretraining_params:
                check_pretraining_params(args.init_pretraining_params, train_program)
                init_pretraining_params(
                    exe,
                    args.init_pretraining_params,
                    main_program=train_program)
    elif args.do_val or args.do_test:
        args.init_checkpoint = args.init_pretraining_params
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        check_pretraining_params(args.init_checkpoint, test_prog)
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=test_prog)

    if args.do_train:
        train_exe = paddle.static.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=graph_vars["loss"].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            main_program=train_program,
            num_trainers=worker_num,
            trainer_id=worker_index)
    else:
        train_exe = None

    if args.do_val or args.do_test:
        test_exe = paddle.static.ParallelExecutor(use_cuda=args.use_cuda,
                                                  main_program=test_prog,
                                                  share_vars_from=train_exe)

    dev_ret_history = []  # (steps, key_eval, eval)
    test_ret_history = []  # (steps, key_eval, eval)
    steps = 0
    if args.do_train:
        train_pyreader.set_batch_generator(train_data_generator, places=place)
        train_pyreader.start()
        time_begin = time.time()
        skip_steps = args.skip_steps
        while True:
            try:
                scheduled_lr.step(steps)
                if steps % skip_steps == 0:
                    train_fetch_list = [graph_vars["loss"].name]
                    res = train_exe.run(fetch_list=train_fetch_list)
                    outputs = {"loss": np.mean(res[0]), 'learning_rate': float(scheduled_lr.get_lr())}
                    if args.verbose:
                        verbose = "train pyreader queue size: %d, learning_rate: %.10f" % \
                                  (train_pyreader.queue.size(), outputs['learning_rate'])
                        print(verbose)
                    current_example, current_epoch = train_data_reader.get_train_progress()
                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("%s - epoch: %d, progress: %d/%d, step: %d, ave loss: %f, speed: %f steps/s" %
                          (get_time(), current_epoch, current_example, num_train_examples,
                           steps, outputs["loss"], args.skip_steps / used_time))
                    time_begin = time.time()
                else:
                    train_exe.run(fetch_list=[])

                if worker_index == 0:
                    if steps % args.save_steps == 0 and args.save_checkpoints and steps > 0:
                        save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                        paddle.fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        test_pyreader.set_batch_generator(dev_data_generator, places=place)
                        outputs = evaluate(args, test_exe, test_prog, test_pyreader, test_graph_vars, "dev",
                                           worker_num, worker_index, data_reader=dev_data_reader)
                        if worker_index == 0:
                            dev_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

                    # evaluate test set
                    if args.do_test:
                        test_pyreader.set_batch_generator(test_data_generator, places=place)
                        outputs = evaluate(args, test_exe, test_prog, test_pyreader, test_graph_vars, "test",
                                           worker_num, worker_index, data_reader=test_data_reader)
                        if worker_index == 0:
                            test_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))

                steps += 1

            except paddle.fluid.core.EOFException:
                if args.save_checkpoints:
                    save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                    paddle.fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        test_pyreader.set_batch_generator(dev_data_generator, places=place)
        if worker_index == 0:
            print("Final validation result:")
        outputs = evaluate(args, test_exe, test_prog, test_pyreader, test_graph_vars, "dev",
                           worker_num, worker_index, data_reader=dev_data_reader)

        if worker_index == 0:
            dev_ret_history.append((steps, outputs['key_eval'], outputs[outputs['key_eval']]))
            dev_ret_history = sorted(dev_ret_history, key=lambda a: a[2], reverse=True)
            print("Best validation result: step %d %s %f" %
                  (dev_ret_history[0][0], dev_ret_history[0][1], dev_ret_history[0][2]))

    # final eval on test set
    if args.do_test:
        test_pyreader.set_batch_generator(test_data_generator, places=place)
        if worker_index == 0:
            print("Final test result:")
        outputs = evaluate(args, test_exe, test_prog, test_pyreader, test_graph_vars, "test",
                           worker_num, worker_index, data_reader=test_data_reader)


if __name__ == '__main__':
    print_arguments(args)
    main(args)
