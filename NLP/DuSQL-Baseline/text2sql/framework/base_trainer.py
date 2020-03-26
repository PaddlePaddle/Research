# -*- coding: utf-8 -*-
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
"""
BaseTrainer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import json
import threading
import multiprocessing
from collections import OrderedDict

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker

import text2sql.framework.utils.init as init
import text2sql.framework.modules.ernie_optimization as opt
from text2sql.framework.rule import InstanceName
from text2sql.framework.utils.util_helper import save_infer_data_meta

TRAIN_STATUS_FILE = 'training_status.json'

class BaseTrainer(object):
    """BaseTrainer"""

    def __init__(self, params, data_set_reader, model_class):
        """
        1.运行环境初始化 2.program初始化 3.计算图网络导入 4.模型参数导入 5.运行(reader) 6.模型导出
        :param params: 运行的基本参数设置
        :param data_set_reader: 运行的基本参数设置
        :param model_class: 使用的是哪个model
        """
        self.data_set_reader = data_set_reader
        self.params = params
        self.model_class = model_class
        self.random_seed = self.params.get("random_seed", 0)
        self.forward_train_output = None
        self.optimizer_output_dict = None
        self.fetch_list_train = []
        self.fetch_list_evaluate = []
        self.is_fleet = False
        self.init_program()
        self.init_env()
        self.init_net()
        self.executor.run(self.startup_program)
        self.prepare_fleet_paddle_cloud(self.is_fleet)
        if self.params["load_checkpoint"] or self.params["load_parameters"]:
            self.load_model_params("net_model")
        elif self.params["pre_train_model"]:
            self.load_model_params("pre_train_model")
        self.build_executor()
        self.curr_epoch = 0
        self.curr_step = 0

    def init_env(self):
        """
        :return:
        """
        # multi nodes
        self.num_trainers = 1
        self.trainer_id = 0
        self.is_local = self.params.get("PADDLE_IS_LOCAL", False)
        # cpu multi
        if self.params["PADDLE_USE_GPU"]:
            gpus = os.getenv('FLAGS_selected_gpus', '0').split(",")
            self.gpu_id = int(gpus[0])
            run_place = fluid.CUDAPlace(int(gpus[0]))
            if "is_distributed" in self.params and self.params["is_distributed"]:
                self.dev_count = len(gpus)
            else:
                self.dev_count = fluid.core.get_cuda_device_count()
            # self.is_local = os.getenv("PADDLE_IS_LOCAL", "0") == "1"
            self.prepare_nccl2_env(self.is_local)
            logging.debug("finish prepare nccl2 env")
        else:
            run_place = fluid.CPUPlace()
            self.dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            # self.is_local = os.getenv("PADDLE_IS_LOCAL", "0") == "1"
            self.prepare_cpumulti_env(self.is_local)
            self.gpu_id = None
            logging.debug("finish prepare cpu multi")
        self.executor = fluid.Executor(run_place)

        # parallel executor relevant config
        self.num_iteration_per_drop_scope = self.params.get("num_iteration_per_drop_scope", 1)
        self.use_fast_executor = self.params.get("use_fast_executor", False)

    def init_program(self):
        """
        :return:
        """
        param_dict = OrderedDict()

        self.startup_program = fluid.Program()
        if self.random_seed is not None:
            self.startup_program.random_seed = self.random_seed
        self.train_program = fluid.Program()
        self.test_program = fluid.Program()
        self.evaluate_program = fluid.Program()
        self.save_inference_program = fluid.Program()

    def load_model_params(self, load_type):
        """
        :return:
        """
        if load_type == "net_model":
            if self.params["load_checkpoint"] and self.params["load_parameters"]:
                raise ValueError(
                    "ERROR: config 'load_checkpoint' and 'load_parameters' "
                    "both are set! Only one of them should be set. "
                    "if you want warmstart checkpoint keep its learning_rate and moments, plese set 'load_checkpoint'. "
                    "if you want warmstart checkpoint with only its parameters, and you want reset a new learning_rate "
                    "by config, please set 'load_parameters'")
            if self.params["load_checkpoint"]:
                init.init_checkpoint(exe=self.executor, init_checkpoint_path=self.params["load_checkpoint"],
                                     main_program=self.startup_program, use_fp16=self.params.get("use_fp16", False))
                self.load_train_status(self.params["load_checkpoint"])
            elif self.params["load_parameters"]:
                init.init_pretraining_params(exe=self.executor, pretraining_params_path=self.params["load_parameters"],
                                             main_program=self.startup_program,
                                             use_fp16=self.params.get("use_fp16", False))
        elif load_type == "pre_train_model":
            # pretrain_embedding_path = self.get_pretrain_embedding_path()
            for pre_train_model in self.params["pre_train_model"]:
                logging.debug("pre_train_model's name = %s" % pre_train_model["name"])
                params_path = pre_train_model["params_path"]
                init.init_pretraining_params(exe=self.executor,
                                             pretraining_params_path=params_path,
                                             main_program=self.startup_program,
                                             use_fp16=self.params.get("use_fp16", False))
        else:
            raise ValueError('load type setting error. expected <net_model|pre_train_model>, but got %s' % (load_type))

    def init_net(self):
        """
        初始化网络
        :return:
        """
        self.init_train_net()
        self.test_program = self.init_evaluate_net(self.data_set_reader.test_reader, self.test_program)
        self.evaluate_program = self.init_evaluate_net(self.data_set_reader.evaluate_reader, self.evaluate_program)
        self.init_save_inference_net()

    def init_train_net(self):
        """
        训练网络初始化，前向+后向
        :return:
        """
        with fluid.program_guard(self.train_program, self.startup_program):
            with fluid.unique_name.guard():
                self.data_set_reader.train_reader.create_reader()
                fields_dict = self.data_set_reader.train_reader.instance_fields_dict()
                self.forward_train_output = self.model_class.forward(fields_dict, phase=InstanceName.TRAINING)

                self.optimizer_output_dict = self.model_class.optimizer(self.forward_train_output[InstanceName.LOSS],
                                                                        self.is_fleet)
                if isinstance(self.optimizer_output_dict, dict):
                    if "use_ernie_opt" in self.optimizer_output_dict:
                        opt_args = self.optimizer_output_dict["opt_args"]
                        self.optimizer_output_dict = opt.optimization(train_program=self.train_program,
                                                                  startup_prog=self.startup_program,
                                                                  **opt_args)
                else:
                    self.optimizer_output_dict = {}
                self.forward_train_output.update(self.optimizer_output_dict)
                self.fetch_list_train = list(self.forward_train_output.values())
                self.fetch_list_train_key = list(self.forward_train_output.keys())

    def init_evaluate_net(self, reader, program):
        """初始化评估过程的网络，网络只有前向
        :return:
        """
        if not reader:
            return None
        with fluid.program_guard(program, self.startup_program):
            with fluid.unique_name.guard():
                reader.create_reader()
                fields_dict = reader.instance_fields_dict()
                self.forward_evaluate_output = self.model_class.forward(fields_dict, phase=InstanceName.EVALUATE)
                self.fetch_list_evaluate = list(self.forward_evaluate_output.values())
                self.fetch_list_evaluate_key = list(self.forward_evaluate_output.keys())
        program = program.clone(for_test=True)
        return program

    def init_save_inference_net(self):
        """初始化用来保存inference model的网络，只有前向，且是裁切过后的网络。
        :return:
        """
        if self.data_set_reader.predict_reader:
            with fluid.program_guard(self.save_inference_program, self.startup_program):
                with fluid.unique_name.guard():
                    self.data_set_reader.predict_reader.create_reader()
                    fields_dict = self.data_set_reader.predict_reader.instance_fields_dict()
                    forward_output_dict = self.model_class.forward(fields_dict, phase=InstanceName.SAVE_INFERENCE)
                    target_feed_list = forward_output_dict[InstanceName.TARGET_FEED_NAMES]
                    self.infer_dict = self.get_infer_data_meta(target_feed_list, fields_dict)
                    self.feed_target_names = target_feed_list
                    self.inference_output = forward_output_dict[InstanceName.TARGET_PREDICTS]
            self.save_inference_program = self.save_inference_program.clone(for_test=True)

    def run(self, phase, need_fetch=True):
        """run and fetch
        :param phase:
        :param need_fetch
        :return:
        """
        fetch_output = []

        if phase == InstanceName.TRAINING:
            fetch_list = self.fetch_list_train if need_fetch else []
            fetch_output = self.train_exe.run(fetch_list=fetch_list)
        elif phase == InstanceName.TEST:
            fetch_output = self.executor.run(program=self.test_program, fetch_list=self.fetch_list_evaluate)
        elif phase == InstanceName.EVALUATE:
            fetch_output = self.executor.run(program=self.evaluate_program, fetch_list=self.fetch_list_evaluate)
        return fetch_output

    def train_and_eval(self):
        """
        :param fetch_list_value:
        :param fetch_list_key:
        :param steps:
        :param phase:
        :return:
        """
        raise NotImplementedError

    def evaluate(self, reader, phase, step):
        """
        :param reader:
        :param phase:
        :param program:
        :param step:
        :return:
        """
        raise NotImplementedError

    def visualdl_log(self, metrics_output, train_loss, steps, phase):
        """log可视化，仅限于paddlecloud 平台任务
        :param metrics_output:
        :param train_loss:
        :param steps:
        :param phase:
        :return:
        """
        logging.debug("{phase} log: steps {steps}, loss {loss}, metrics: {metrics}".format(
                      phase=phase, steps=steps, loss=train_loss, metrics=metrics_output))
        try:
            if metrics_output and len(metrics_output) != 0:
                import paddlecloud.visual_util as visualdl
                x_dic = {"x_name": "step", "x_value": steps}
                y_ls = []
                for key, value in metrics_output.items():
                    y = {}
                    y["y_name"] = key
                    y["y_value"] = value
                    y_ls.append(y)
                visualdl.show_fluid_trend(x_dic, y_ls, tag=phase)
        except Exception:
            logging.error("import paddlecloud.visual_util failed")

    def build_executor(self):
        """
        :return:
        """
        if self.is_fleet:
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = self.dev_count
            build_strategy = fluid.BuildStrategy()
            build_strategy.async_mode = False
            logging.info("CPU_NUM = %d" % self.dev_count)
            if self.dev_count > 1:
                build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            self.train_exe = fluid.ParallelExecutor(
                use_cuda=self.params["PADDLE_USE_GPU"],
                loss_name=self.forward_train_output[InstanceName.LOSS].name,
                main_program=self.train_program,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
        else:
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = self.dev_count
            exec_strategy.num_iteration_per_drop_scope = self.num_iteration_per_drop_scope
            if self.use_fast_executor:
                exec_strategy.use_experimental_executor = True
            self.train_exe = fluid.ParallelExecutor(
                use_cuda=self.params["PADDLE_USE_GPU"],
                loss_name=self.forward_train_output[InstanceName.LOSS].name,
                exec_strategy=exec_strategy,
                main_program=self.train_program,
                num_trainers=self.num_trainers,
                trainer_id=self.trainer_id)

    def prepare_cpumulti_env(self, is_local):
        """
        :param is_local:
        :return:
        """
        if is_local:
            self.is_fleet = False
        else:
            role = role_maker.PaddleCloudRoleMaker()
            fleet.init(role)
            self.is_fleet = True
            logging.debug("init fleet cpu multi")

    def prepare_fleet_paddle_cloud(self, is_fleet):
        """
        :param is_local:
        :return:
        """
        if is_fleet == False:
            self.executor.run(self.startup_program)
        else:
            if fleet.is_worker():
                self.trainer_id = fleet.worker_index()
            if fleet.is_server():
                logging.info("init and run fleet server")
                fleet.init_server()
                fleet.run_server()
            elif fleet.is_worker():
                logging.info("init and run fleet worker")
                fleet.init_worker()
                self.executor.run(self.startup_program)

    def prepare_nccl2_env(self, is_local):
        """
        :param is_local:
        :return:
        """
        if not is_local:
            logging.debug("is_distributed: %s" % self.params["is_distributed"])
            if self.params["is_distributed"]:
                trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
                worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
                current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
                worker_endpoints = worker_endpoints_env.split(",")
                trainers_num = len(worker_endpoints)
                logging.debug("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
                      trainer_id:{}".format(worker_endpoints, trainers_num,
                                            current_endpoint, trainer_id))
                # prepare nccl2 env.
                config = fluid.DistributeTranspilerConfig()
                config.mode = "nccl2"
                t = fluid.DistributeTranspiler(config=config)
                t.transpile(
                    trainer_id,
                    trainers=worker_endpoints_env,
                    current_endpoint=current_endpoint,
                    program=self.train_program if self.params["is_do_train"] else self.test_program,
                    startup_program=self.startup_program)
                self.num_trainers = trainers_num
                self.trainer_id = trainer_id

    def get_infer_data_meta(self, target_feed_list, fields_dict):
        """
        :param target_feed_list:
        :param fields_dict:
        :return:
        """
        infer_dict = {
            "fields": []
        }
        for name in target_feed_list:
            for k1, v1 in fields_dict.items():  # dict_keys(['text_a', 'label'])
                for k2, v2 in v1.items():
                    if v2:
                        for k3 in v2:
                            if v2[k3] and v2[k3].name == name:
                                field_ele = "%s#%s" % (k1, k3)
                                infer_dict["fields"].append(field_ele)
        return infer_dict

    def save_models(self, save_checkpoints_path, save_inference_path, steps, save_checkpoint=True, save_inference=True):
        """
        :param save_checkpoints_path:
        :param save_inference_path:
        :param steps:
        :return:
        """
        if save_checkpoint:
            self.save_checkpoint(self.executor, save_checkpoints_path, self.train_program, steps)
        if save_inference:
            self.save_inference(self.executor, self.feed_target_names, self.inference_output,
                                save_inference_path, self.save_inference_program,
                                self.train_program, steps, self.infer_dict)

    def save_checkpoint(self, exe, save_checkpoints_path, program, steps):
        """
        :param exe:
        :param save_checkpoints_path:
        :param program:
        :param steps:
        :return:
        """
        save_path = os.path.join(save_checkpoints_path, "checkpoints_step_" + str(steps))
        if self.is_fleet:
            logging.info("fleet save checkpoints")
            fleet.save_persistables(exe, save_path, program)
        else:
            fluid.io.save_persistables(exe, save_path, program)

        dct_train_status = {'epoch': self.curr_epoch,
                            'step': self.curr_step}
        status_file = os.path.join(save_path, TRAIN_STATUS_FILE)
        with open(status_file, 'w') as ofs:
            json.dump(dct_train_status, ofs)

    def save_inference(self, exe, feeded_var_names, target_vars, save_inference_path,
                       program, main_program, steps, data_dict):
        """
        :param exe:
        :param feeded_var_names
        :param target_vars
        :param save_inference_path:
        :param program:
        :param steps:
        :param data_dict:
        :return:
        """
        save_path = os.path.join(save_inference_path, "inference_step_" + str(steps))
        if self.is_fleet:
            logging.info("fleet save models")
            fleet.save_inference_model(
                executor=exe,
                dirname=save_path,
                feeded_var_names=feeded_var_names,
                target_vars=target_vars,
                main_program=main_program)
        else:
            fluid.io.save_inference_model(
                save_path,
                feeded_var_names,
                target_vars,
                exe,
                main_program=program,
                model_filename="model",
                params_filename="params")
        save_infer_data_meta(data_dict, save_path + '/infer_data_params.json')

    def load_train_status(self, checkpoint_dir):
        """init training status by checkpoint

        Args:
            checkpoint_dir (TYPE): NULL

        Returns: TODO

        Raises: NULL
        """
        status_file = os.path.join(checkpoint_dir, TRAIN_STATUS_FILE)
        if not os.path.isfile(status_file):
            logging.warn('training status file is lost: %s', status_file)
            self.curr_epoch = 0
            self.curr_step = 0
        else:
            with open(status_file) as ifs:
                dct_train_status = json.load(ifs)
            self.curr_epoch = dct_train_status.get('curr_epoch', 0)
            self.curr_step = dct_train_status.get('curr_step', 0)
            logging.info('training status is inited: epoch=%d, step=%d', self.curr_epoch, self.curr_step)

