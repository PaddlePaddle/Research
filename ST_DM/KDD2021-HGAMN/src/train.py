#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""listwise model
"""

import geohash
import os
import re
import time
import logging
from random import random
from functools import reduce, partial

import numpy as np
import multiprocessing

import paddle
import paddle.fluid as F
import paddle.fluid.layers as L
import pgl

from utils.args import print_arguments, check_cuda, prepare_logger
from utils.init import init_checkpoint, init_pretraining_params
from finetune_args import parser
from lbs_model import BaseGraphErnie, GCNGraphErnie, GATGraphErnie
import lbs_model
from dataset.listwise_dataset import DataGenerator, GraphDataGenerator
from optimization import optimization
from monitor.listwise_monitor import train_and_evaluate

log = logging.getLogger(__name__)


class Metric(object):
    """Metric"""

    def __init__(self, **args):
        self.args = args

    @property
    def vars(self):
        """ fetch metric vars"""
        values = [self.args[k] for k in self.args.keys()]
        return values

    def parse(self, fetch_list):
        """parse"""
        tup = list(zip(self.args.keys(), [float(v[0]) for v in fetch_list]))
        return dict(tup)


if __name__ == '__main__':
    args = parser.parse_args()
    prepare_logger(log)
    print_arguments(args)

    train_prog = F.Program()
    startup_prog = F.Program()

    if args.use_cuda:
        dev_list = F.cuda_places()
        place = dev_list[0]
        dev_count = len(dev_list)
    else:
        place = F.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    with F.program_guard(train_prog, startup_prog):
        with F.unique_name.guard():
            if args.sage_mode == "gcn":
                graph_ernie = GCNGraphErnie(args, task="listwise")
            elif args.sage_mode == "gat":
                graph_ernie = GATGraphErnie(args, task="listwise")
            elif args.sage_mode == "gatne":
                graph_ernie = lbs_model.GATNEGraphErnie(args, task="listwise")
            elif args.sage_mode == "hgcmn":
                graph_ernie = lbs_model.HGCMNGraphErnie(args, task="listwise")
            elif args.sage_mode == "base":
                graph_ernie = BaseGraphErnie(args, task="listwise")
            else:
                raise ValueError("Mode not found %s" % args.sage_mode)

            train_ds = GraphDataGenerator(
                graph_path=args.graph_data,
                data_paths=[
                    args.data_path,
                ],
                graph_wrapper=graph_ernie.graph_wrapper,
                vocab_path=args.vocab_path,
                avoid_leak=False,
                num_workers=args.num_workers,
                max_seq_len=args.max_seq_len,
                token_mode=args.token_mode,
                batch_size=args.batch_size)

            num_train_examples = len(train_ds)

            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count
            warmup_steps = int(max_train_steps * args.warmup_proportion)
            scheduled_lr, loss_scaling = optimization(
                loss=graph_ernie.loss,
                warmup_steps=warmup_steps,
                num_train_steps=max_train_steps,
                learning_rate=args.learning_rate,
                train_program=train_prog,
                startup_prog=startup_prog,
                weight_decay=args.weight_decay,
                scheduler=args.lr_scheduler,
                use_fp16=False,
                use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                init_loss_scaling=args.init_loss_scaling,
                incr_every_n_steps=args.incr_every_n_steps,
                decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
                incr_ratio=args.incr_ratio,
                decr_ratio=args.decr_ratio)

    test_prog = F.Program()
    with F.program_guard(test_prog, startup_prog):
        with F.unique_name.guard():
            if args.sage_mode == "gcn":
                _graph_ernie = GCNGraphErnie(args, task="listwise")
            elif args.sage_mode == "gat":
                _graph_ernie = GATGraphErnie(args, task="listwise")
            elif args.sage_mode == "gatne":
                _graph_ernie = lbs_model.GATNEGraphErnie(args, task="listwise")
            elif args.sage_mode == "hgcmn":
                _graph_ernie = lbs_model.HGCMNGraphErnie(args, task="listwise")
            elif args.sage_mode == "base":
                _graph_ernie = BaseGraphErnie(args, task="listwise")
            else:
                raise ValueError("Mode not found %s" % args.sage_mode)

    test_prog = test_prog.clone(for_test=True)

    valid_ds = GraphDataGenerator(
        graph_path=args.graph_data,
        data_paths=[args.eval_path],
        graph_wrapper=graph_ernie.graph_wrapper,
        num_workers=args.num_workers,
        avoid_leak=False,
        vocab_path=args.vocab_path,
        max_seq_len=args.max_seq_len,
        token_mode=args.token_mode,
        batch_size=args.batch_size)

    exe = F.Executor(place)
    exe.run(startup_prog)

    if args.init_pretraining_params is not None:
        init_pretraining_params(
            exe, args.init_pretraining_params, main_program=startup_prog)

    metric = Metric(loss=graph_ernie.metrics[0], top1=graph_ernie.acc)

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    if dev_count > 1:

        exec_strategy = F.ExecutionStrategy()
        exec_strategy.num_threads = dev_count

        train_exe = F.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=graph_ernie.loss.name,
            exec_strategy=exec_strategy,
            main_program=train_prog,
            num_trainers=nccl2_num_trainers,
            trainer_id=nccl2_trainer_id)

        test_exe = exe
    else:
        train_exe, test_exe = exe, exe

    train_and_evaluate(
        exe=exe,
        train_exe=train_exe,
        valid_exe=test_exe,
        train_ds=train_ds,
        valid_ds=valid_ds,
        train_prog=train_prog,
        valid_prog=test_prog,
        train_log_step=100,
        output_path=args.output_path,
        dev_count=dev_count,
        model=graph_ernie,
        epoch=args.epoch,
        eval_step=10000,
        metric=metric)
