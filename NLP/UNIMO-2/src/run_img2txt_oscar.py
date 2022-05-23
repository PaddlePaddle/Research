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
File: run_img2txt_oscar.py
Author: liwei(liwei85@baidu.com)
Date: 2021-10-25 15:41
Desc: Finetuning on image-to-text generation tasks.
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

from reader.img2txt_oscar_reader import Img2TxtReader
from model.roberta_tokenization import GptBpeTokenizer
from model.unimo_grounded import VlConfig
from utils.optimization_fleet import optimization
from utils.init import init_model, check_pretraining_params
from utils.args import print_arguments
from utils.utils import visualdl_log
from finetune.img2txt_oscar import Img2Txt
from args.img2txt_oscar_args import parser
from functools import partial
from collections import OrderedDict

args = parser.parse_args()


def get_time():
    res = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return res


def evaluate_datasets(pyreader, reader, eval_func, data_generator,
                      do_pred=False, suffix="out"):
    def evaluate_dataset(phase, path):
        pyreader.set_batch_generator(data_generator(filelist=path, phase=phase))
        eval_func(eval_phase="%s_%s" % (phase, suffix))

    if args.do_val:
        evaluate_dataset("dev", args.valid_filelist)
    if args.do_test:
        evaluate_dataset("test", args.test_filelist)
    if args.do_pred and do_pred:
        evaluate_dataset("pred", args.test_filelist)


def save_checkpoint(program, exe, suffix):
    save_path = os.path.join(args.checkpoints, suffix)
    paddle.fluid.io.save_persistables(exe, save_path, program)


def main(args):
    vl_config = VlConfig(args.vl_config_path)
    if args.hidden_dropout_prob >= 0:
        vl_config["hidden_dropout_prob"] = args.hidden_dropout_prob
    if args.attention_probs_dropout_prob >= 0:
        vl_config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob

    vl_config['image_size'] = args.image_size
    vl_config["resolution"] = args.resolution
    vl_config['num_codebook'] = args.num_codebook
    vl_config.print_config()

    if args.pred_batch_size <= 0:
        args.pred_batch_size = args.batch_size

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

    """load vocabulary"""
    tokenizer = GptBpeTokenizer(vocab_file=args.roberta_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=True)

    reader = Img2TxtReader(tokenizer=tokenizer,
                           args=args,
                           image_size=args.image_size,
                           resolution=args.resolution)
    img2txt = Img2Txt(args, vl_config, tokenizer)

    if not (args.do_train or args.do_val or args.do_test or args.do_pred):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = paddle.static.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        paddle.seed(args.random_seed)

    if args.do_train:
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
        train_data_generator = reader.data_generator(
            filelist=args.train_filelist,
            batch_size=args.batch_size,
            epoch=args.epoch,
            dev_count=trainers_num,
            shuffle=True,
            phase="train")

        num_train_examples = 566747  # reader.get_num_examples(args.train_filelist)
        max_train_steps = args.epoch * num_train_examples // args.batch_size // trainers_num

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_prog):
            train_pyreader, graph_vars, checkpoints = img2txt.create_model()

            if args.use_recompute:
                dist_strategy.recompute_configs = {"checkpoints": checkpoints}

            scheduled_lr = optimization(
                loss=graph_vars["loss"],
                warmup_steps=warmup_steps,
                num_train_steps=max_train_steps,
                learning_rate=args.learning_rate,
                train_program=train_program,
                weight_decay=args.weight_decay,
                scheduler=args.lr_scheduler,
                beta1=args.beta1,
                beta2=args.beta2,
                epsilon=args.epsilon,
                dist_strategy=dist_strategy)

    if args.do_val or args.do_test or args.do_pred:
        test_prog = paddle.static.Program()
        with paddle.static.program_guard(test_prog, startup_prog):
            test_pyreader, test_graph_vars = img2txt.create_model(decoding=args.do_decode)
        test_prog = test_prog.clone(for_test=True)

    exe = paddle.static.Executor(place)
    exe.run(startup_prog)

    if args.init_checkpoint:
        check_pretraining_params(args.init_checkpoint, train_program if args.do_train else test_prog)
    elif args.init_pretraining_params:
        check_pretraining_params(args.init_pretraining_params, train_program if args.do_train else test_prog)
    else:
        print("Note!!!!!!!!!!!!!!")
        print("No pretraining_params are setted. Run from random params....")

    init_model(args, exe, train_program if args.do_train else test_prog)

    if args.do_train:
        train_exe = paddle.static.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=graph_vars["loss"].name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            main_program=train_program,
            num_trainers=worker_num,
            trainer_id=worker_index)
        train_pyreader.set_batch_generator(train_data_generator)
        train_resource = {"exe": train_exe,
                          "program": train_program,
                          "pyreader": train_pyreader}
        save_model = partial(save_checkpoint, program=train_program, exe=exe)

    test_dev_count = 1
    if args.do_val or args.do_test or args.do_pred:
        test_exe = exe
        if args.use_multi_gpu_test:
            test_dev_count = worker_num
        test_resource = {"exe": test_exe,
                         "program": test_prog,
                         "pyreader": test_pyreader}
        eval_data_generator = partial(reader.data_generator, batch_size=args.pred_batch_size,
                                      epoch=1, dev_count=test_dev_count, shuffle=False, do_decode=args.do_decode,
                                      place=place)
        eval_func = partial(img2txt.evaluate, resource=test_resource, graph_vars=test_graph_vars,
                            dev_count=test_dev_count, output_path=args.checkpoints, gpu_id=worker_index)
        evaluate = partial(evaluate_datasets, pyreader=test_pyreader, reader=reader,
                           eval_func=eval_func, data_generator=eval_data_generator)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        last_epoch = 0
        time_begin = time.time()
        skip_steps = args.skip_steps
        while True:
            try:
                steps += 1
                scheduled_lr.step(steps - 1)
                if args.save_and_valid_by_epoch:
                    suffix = "epoch_" + str(last_epoch)
                else:
                    suffix = "step_" + str(steps)

                if steps % skip_steps == 0:
                    outputs = img2txt.evaluate(train_resource, "train", graph_vars)
                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size()
                        verbose += "learning rate: %.8f" % (float(scheduled_lr.get_lr()))
                        print(verbose)

                    current_epoch = steps * args.batch_size * trainers_num // num_train_examples
                    current_example = steps * args.batch_size * trainers_num % num_train_examples

                    time_end = time.time()
                    used_time = time_end - time_begin
                    print("%s - epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                          "ppl: %f, speed: %f steps/s"
                          % (get_time(), current_epoch, current_example, num_train_examples,
                             steps, outputs["loss"], outputs["ppl"],
                             args.skip_steps / used_time))
                    time_begin = time.time()

                    if args.visualdl_log and worker_index == 0:
                        visuallog_dict = OrderedDict()
                        visuallog_dict["ppl"] = outputs["ppl"]
                        visualdl_log(visuallog_dict, outputs["ppl"], steps, phase='train')
                else:
                    train_exe.run(fetch_list=[])

                if worker_index >= test_dev_count:
                    continue

                do_save = False
                do_eval = False
                if not args.save_and_valid_by_epoch:
                    if steps % args.save_steps == 0 and worker_index == 0:
                        do_save = True
                    if steps % args.validation_steps == 0:
                        do_eval = True
                else:
                    current_epoch = steps * args.batch_size * trainers_num // num_train_examples
                    if current_epoch != last_epoch:
                        if worker_index == 0:
                            do_save = True
                        do_eval = True

                if do_save:
                    save_model(suffix=suffix)
                if do_eval:
                    evaluate(suffix=suffix)

                if args.save_and_valid_by_epoch:
                    last_epoch = current_epoch

            except paddle.fluid.core.EOFException:
                save_model(suffix=suffix)
                train_pyreader.reset()
                break

    if worker_index >= test_dev_count:
        return

    if args.do_val or args.do_test or args.do_pred:
        suffix = "output"
        if args.do_train:
            if not args.save_and_valid_by_epoch:
                suffix = "step_" + str(steps)
            else:
                suffix = "epoch_" + str(last_epoch)

        evaluate(suffix=suffix, do_pred=True)


if __name__ == '__main__':
    print_arguments(args)
    main(args)
