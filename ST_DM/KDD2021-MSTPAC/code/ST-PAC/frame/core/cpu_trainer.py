#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################


"""
 Specify the brief cpu_trainer.py
 Date: 2019/07/24 17:53:40
 Brief:
 CPUTrainer for cpu platform training. 
 Please set FLAGS.num_preprocessing_threads for multi-core training.
 User can data_reader as dataset, pyreader, datafeed.
"""

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import logging
import shutil

import paddle.fluid as fluid

from base_frame import BaseFrame


class CPUTrainer(BaseFrame):
    """
    CPU datafeed CPUTrainer
    CPUTrainer converts the data that returned by a reader into 
    a data structure that can feed into Executor
    """

    def __init__(self):
        super(CPUTrainer, self).__init__()
        self.ckpt_list = []
    
    def record_checkpoint(self, FLAGS, global_step):
        """
            record checkpoint
            TODO: restore checkpoint, distribute env
        """
        ckpt_path = "%s/checkpoint.meta" % (FLAGS.train_dir)
        path = "%s/checkpoint_%s" % (FLAGS.train_dir, global_step)
        self.ckpt_list.append(path)
        logging.info("save_max_to_keep: %d, current ckpt_list: %s", FLAGS.save_max_to_keep, 
                str(self.ckpt_list))
        if len(self.ckpt_list) > FLAGS.save_max_to_keep:
            ckpt_to_keep = self.ckpt_list[-FLAGS.save_max_to_keep:]
            ckpt_to_remove = set(self.ckpt_list) - set(ckpt_to_keep)
            self.ckpt_list = ckpt_to_keep
            for ckpt in ckpt_to_remove:
                ckpt_dir = ckpt
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)

        ckpt_file = open(ckpt_path, "w")
        i = 1
        ckpt_file_content = "[Monitor]" + "\n" + "ckpt_version: " + str(global_step) + "\n"
        for ckpt in self.ckpt_list:
            if i == len(self.ckpt_list):
                ckpt_str = "init_pretrain_model: " + ckpt + "\n"
                ckpt_str += "init_train_params: None\n"
                ckpt_str += "eval_dir: " + ckpt + "\n"
            else:
                ckpt_str = "ckpt_" + str(i) + ": " + ckpt + "\n"
            i += 1
            ckpt_file_content += ckpt_str
        ckpt_file.write(ckpt_file_content)
        ckpt_file.close()

    def save_model(self, FLAGS, net_output, global_step):
        """
            save model
        """
 
        if global_step != "final" and global_step % FLAGS.save_model_steps != 0:
            return

        path = "%s/checkpoint_%s" % (FLAGS.train_dir, global_step)
        fluid.io.save_inference_model(path, 
                   net_output['model_output']['feeded_var_names'],
                   net_output["model_output"]['fetch_targets'],
                   self.paddle_env['exe'], self.get_infer_program(FLAGS), program_only=True)
        #fluid.io.save_params(self.paddle_env['exe'], path)
        fluid.io.save_persistables(self.paddle_env['exe'], path)
        self.record_checkpoint(FLAGS, global_step)

    def train(self, FLAGS, net_output):
        """
        start training
        """
        program = self.get_main_program(FLAGS)
        self.init_model_params(self.paddle_env['exe'], program, FLAGS)
        net_instance = self.paddle_env["factory"]["net"]
         
        if not isinstance(program, fluid.CompiledProgram) and FLAGS.platform == 'local-cpu' \
                and FLAGS.data_reader in ("pyreader", "async"):
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = self.get_thread_num(FLAGS)  #2 for fp32 4 for fp16, CPU_NUM
            exec_strategy.use_experimental_executor = True
            exec_strategy.num_iteration_per_drop_scope = 10  #important shit

            build_strategy = fluid.BuildStrategy()
            build_strategy.remove_unnecessary_lock = False
            build_strategy.enable_inplace = True
            #build_strategy.fuse_broadcast_ops = True
            #build_strategy.memory_optimize = True
            #if exec_strategy.num_threads > 1:
                #build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            program = fluid.CompiledProgram(program).with_data_parallel(
                    loss_name=net_output["loss"].name,
                    build_strategy=build_strategy,
                    exec_strategy=exec_strategy)
        
        global_step = 0
        for epoch_id in range(FLAGS.num_epochs_input):
            batch_id = 0
            if FLAGS.data_reader == "dataset":
                #ONLY for: train time very short in one step
                self.paddle_env['exe'].train_from_dataset(program=program,
                        dataset=self.paddle_env['dataset'],
                        fetch_list=self.debug_tensors,
                        fetch_info=self.debug_keys,
                        print_period=FLAGS.log_every_n_steps,
                        debug=FLAGS.debug_mode)
                #FIXME: dataset not support batch level step
                #net_instance.train_format(None, global_step, epoch_id, batch_id)
                global_step += 1
                self.save_model(FLAGS, net_output, global_step)
            elif (FLAGS.data_reader == "async" or FLAGS.data_reader == "pyreader") \
                and not FLAGS.py_reader_iterable:
                #Before the start of every epoch, call start() to invoke PyReader;
                self.paddle_env["data_reader"].start()
                try:
                    while True:
                        result = self.paddle_env["exe"].run(program=program,
                                         fetch_list=self.debug_tensors,
                                         return_numpy=False)
                        net_instance.train_format(result, global_step, epoch_id, batch_id)
                        global_step += 1
                        batch_id += 1
                        if FLAGS.max_number_of_steps is not None and global_step >= FLAGS.max_number_of_steps:
                            break
                        self.save_model(FLAGS, net_output, global_step)
                except fluid.core.EOFException:
                    """
                    At the end of every epoch, read_file throws exception 
                    fluid.core.EOFException . Call reset() after catching up 
                    exception to reset the state of PyReader in order to start next epoch.
                    """
                    self.paddle_env["data_reader"].reset() 
            else:
                for sample in self.paddle_env["data_reader"]():
                    if self.paddle_env["data_feeder"] is not None:
                       sample = self.paddle_env["data_feeder"].feed(sample) 
                    result = self.paddle_env["exe"].run(program=program, 
                        feed=sample,
                        fetch_list=self.debug_tensors,
                        return_numpy=False)
                    net_instance.train_format(result, global_step, epoch_id, batch_id)
                    global_step += 1
                    batch_id += 1 
                    if FLAGS.max_number_of_steps is not None and global_step >= FLAGS.max_number_of_steps:
                        break
                    self.save_model(FLAGS, net_output, global_step)
        
            if FLAGS.max_number_of_steps is not None and global_step >= FLAGS.max_number_of_steps:
                break

        self.save_model(FLAGS, net_output, "final")

    def init_model_params(self, exe, main_program, FLAGS):
        """
            load params of pretrained model, NOT including moment, learning_rate
        """
        if FLAGS.init_train_params is not None:
            place = self.create_places(FLAGS)[0]
            self.paddle_env['factory']['net'].init_params(place)
            logging.info("Load pretrain params from {}.".format(
                 FLAGS.init_train_params))
        elif FLAGS.init_pretrain_model is not None:
            fluid.io.load_persistables(
                exe,
                FLAGS.init_pretrain_model,
                main_program=main_program)
            logging.info("Load pretrain persistables from {}.".format(
                 FLAGS.init_pretrain_model))
        return


if __name__ == '__main__':
    trainer = CPUTrainer()
    ret = trainer.start(sys.argv)
    if not ret:
        sys.exit(-1)
 
