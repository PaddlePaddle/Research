#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""train process of text2sql task
"""

import sys
import os
import traceback
import logging
import time
import json
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
from paddle import fluid
from paddle.fluid.incubate.fleet.collective import fleet
from text2sql.framework import register
from text2sql.framework.rule import InstanceName as C
from text2sql.framework.base_trainer import BaseTrainer

from text2sql import cli_args
from text2sql.datalib import json_dataset
from text2sql.models.text2sql_model import Text2SQL


class Text2SQLTrainer(BaseTrainer):

    """Text2SQL Trainner"""

    def __init__(self, params, dataset_reader, model_obj):
        """init of trainer

        Args:
            params (TYPE): NULL
            dataset_reader (TYPE): NULL
            model_obj (TYPE): NULL
        """
        super(Text2SQLTrainer, self).__init__(params, dataset_reader, model_obj)
        self.save_eval_result = 'save_predict_file' in params

    def train_and_eval(self):
        """
        Returns: None
        """
        if self.is_fleet and fleet.is_server():
            logging.debug("is fleet.server, over")
            return
        if self.is_fleet:
            logging.debug("worker_index%d start train...." % fleet.worker_index())

        num_train_examples = self.params.get("num_train_examples", 0)
        if num_train_examples == 0:
            num_train_examples = self.data_set_reader.train_reader.get_num_examples()

        self.data_set_reader.train_reader.run()
        self.curr_step = 1
        time_begin = time.time()
        if 'output_path' in self.params and self.params["output_path"]:
            save_checkpoints_path = os.path.join(self.params["output_path"], "save_checkpoints")
            save_inference_model_path = os.path.join(self.params["output_path"], "save_inference_model")
        else:
            save_checkpoints_path = "./output/save_checkpoints/"
            save_inference_model_path = "./output/save_inference_model/"
        while True:
            try:
                if self.curr_step % self.params["train_log_step"] != 0:
                    self.run(C.TRAINING, need_fetch=False)
                else:
                    metrics_tensor_value = self.run(C.TRAINING, need_fetch=True)
                    current_example, self.curr_epoch = self.data_set_reader.train_reader.get_train_progress()
                    logging.debug("epoch {%d} progress {%d}/{%d} pyreader queue size {%d}",
                                  self.curr_epoch, current_example, num_train_examples,
                                  self.data_set_reader.train_reader.paddle_py_reader.queue.size())

                    fetch_output_dict = OrderedDict()
                    for key, value in zip(self.fetch_list_train_key, metrics_tensor_value):
                        fetch_output_dict[key] = value
                    time_end = time.time()
                    used_time = time_end - time_begin
                    meta_info = OrderedDict()
                    meta_info[C.STEP] = self.curr_step
                    meta_info[C.GPU_ID] = self.gpu_id
                    meta_info[C.TIME_COST] = used_time
                    meta_info["epoch"] = self.curr_epoch

                    metrics_output = self.model_class.get_metrics(fetch_output_dict, meta_info, C.TRAINING)
                    if self.params.get("visualdl_log", False):
                        assert isinstance(metrics_output, OrderedDict), "metrics_output is must be OrderedDict"
                        self.visualdl_log(
                                metrics_output, np.mean(fetch_output_dict[C.LOSS]), self.curr_step, phase=C.TRAINING)

                if self.trainer_id == 0 and self.curr_step % self.params["save_model_step"] == 0:
                    self.save_models(save_checkpoints_path, save_inference_model_path, self.curr_step)
                if self.curr_step % self.params["eval_step"] == 0:
                    if self.params["is_eval_dev"]:
                        self.evaluate(self.data_set_reader.evaluate_reader, C.EVALUATE, self.curr_step)
                    if self.params["is_eval_test"]:
                        self.evaluate(self.data_set_reader.test_reader, C.TEST, self.curr_step)
                if self.curr_step % self.params["train_log_step"] == 0:
                    time_begin = time.time()
                self.curr_step += 1
                if "steps_for_test" in self.params and self.curr_step >= self.params["steps_for_test"]:
                    self.data_set_reader.train_reader.stop()
                    logging.debug("steps_for_test stop!")
                    break
            except fluid.core.EOFException:
                self.data_set_reader.train_reader.stop()
                break
            except Exception as e:
                logging.error('traceback.format_exc(): %s', traceback.format_exc())
                self.save_models(save_checkpoints_path, save_inference_model_path, self.curr_step)
                raise e
        if self.params["is_eval_dev"]:
            logging.info("Final evaluate result")
            self.evaluate(self.data_set_reader.evaluate_reader, C.EVALUATE, self.curr_step)
        if self.params["is_eval_test"]:
            logging.info("Final test result")
            self.evaluate(self.data_set_reader.test_reader, C.TEST, self.curr_step)

        self.save_models(save_checkpoints_path, save_inference_model_path, self.curr_step)

    def evaluate(self, reader, phase, step):
        """

        Args:
            reader (TYPE): NULL
            phase (TYPE): NULL
            step (TYPE): NULL

        Returns: None
        """
        if not reader:
            raise ValueError("{0} reader is none".format(phase))
        reader.run()
        all_metrics_tensor_value = None
        time_begin = time.time()
        while True:
            try:
                metrics_tensor_value = self.run(phase=phase)
                if all_metrics_tensor_value is None:
                    all_metrics_tensor_value = [[tensor] for tensor in metrics_tensor_value]
                    continue

                for j in range(len(metrics_tensor_value)):
                    all_metrics_tensor_value[j].append(metrics_tensor_value[j])
            except fluid.core.EOFException:
                reader.stop()
                break

        fetch_output_dict = OrderedDict()
        for key, value in zip(self.fetch_list_evaluate_key, all_metrics_tensor_value):
            fetch_output_dict[key] = value
        time_end = time.time()
        used_time = time_end - time_begin

        if self.save_eval_result:
            self.model_class.parse_predict_result(fetch_output_dict[C.PREDICT_RESULT])

        meta_info = OrderedDict()
        meta_info[C.STEP] = step
        meta_info[C.GPU_ID] = self.gpu_id
        meta_info[C.TIME_COST] = used_time
        meta_info["epoch"] = self.curr_epoch
        metrics_output = self.model_class.get_metrics(fetch_output_dict, meta_info, phase)
        if self.params.get("visualdl_log", False):
            assert isinstance(metrics_output, OrderedDict), "the metrics_output must be OrderedDict"
            eval_loss = np.mean(fetch_output_dict[C.LOSS])
            self.visualdl_log(metrics_output, eval_loss, step, phase=phase)


if __name__ == "__main__":
    args = cli_args.init_args()
    logging.basicConfig(level=logging.INFO,
        format="%(levelname)s: %(asctime)s %(filename)s"
        " [%(funcName)s:%(lineno)d][%(process)d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
        filename=args.log_file,
        filemode='a')

    try:
        param_dict = cli_args.init_config(args, args.config, args.db_max_len)
        register.import_modules()

        dataset = json_dataset.T2SDataSet(param_dict["dataset_reader"])
        model = Text2SQL(param_dict["model"], param_dict["trainer"].get('save_predict_file', None))
        trainer = Text2SQLTrainer(param_dict["trainer"], dataset, model)
        trainer.train_and_eval()
    except Exception as e:
        traceback.print_exc()
        if args.log_file is not None:
            logging.critical(traceback.format_exc())
        exit(-1)

