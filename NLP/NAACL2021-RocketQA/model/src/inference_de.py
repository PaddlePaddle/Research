#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import logging
import multiprocessing
import numpy as np

# NOTE(paddle-dev): All of these flags should be
# set before `import paddle`. Otherwise, it would
# not take any effect.
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid

import reader.reader_de_infer as reader_de_infer 
from model.ernie import ErnieConfig
from finetune.dual_encoder_infer import create_model, predict
from utils.args import print_arguments, check_cuda, prepare_logger
from utils.init import init_pretraining_params, init_checkpoint
from finetune_args import parser

args = parser.parse_args()
log = logging.getLogger()

def main(args):
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    if args.use_cuda:
        dev_list = fluid.cuda_places()
        place = dev_list[0]
        dev_count = len(dev_list)
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    exe = fluid.Executor(place)

    reader = reader_de_infer.ClassifyReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        q_max_seq_len=args.q_max_seq_len,
        p_max_seq_len=args.p_max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed,
        tokenizer=args.tokenizer,
        for_cn=args.for_cn,
        task_id=args.task_id)

    assert args.test_save is not None
    startup_prog = fluid.Program()

    test_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            test_pyreader, graph_vars = create_model(
                args,
                pyreader_name='test_reader',
                ernie_config=ernie_config,
                batch_size=args.batch_size,
                is_prediction=True)

    test_prog = test_prog.clone(for_test=True)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if not args.init_checkpoint:
        raise ValueError("args 'init_checkpoint' should be set if"
                        "only doing validation or testing!")
    init_checkpoint(
        exe,
        args.init_checkpoint,
        main_program=startup_prog)

    test_sets = args.test_set.split(',')
    save_dirs = args.test_save.split(',')
    assert len(test_sets) == len(save_dirs)
    batch_size = args.batch_size if args.predict_batch_size is None else args.predict_batch_size

    for test_f, save_f in zip(test_sets, save_dirs):
        test_pyreader.decorate_tensor_provider(
            reader.data_generator(
                test_f,
                batch_size=batch_size,
                epoch=1,
                dev_count=1,
                shuffle=False))

        save_path = save_f
        log.info("testing {}, save to {}".format(test_f, save_path))
        predict(
            args,
            exe,
            test_prog,
            test_pyreader,
            graph_vars,
            output_item=args.output_item,
            output_file_name=args.output_file_name,
            hidden_size=ernie_config['hidden_size'])

if __name__ == '__main__':
    prepare_logger(log)
    print_arguments(args)
    check_cuda(args.use_cuda)
    main(args)
