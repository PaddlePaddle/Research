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

"""command line arguments and some utils
"""

import sys
import os
import traceback
import logging
import argparse
from pathlib import Path
import json

from text2sql import g
from text2sql.grammar import Grammar

def create_arg_parser():
    """create command line arguments parser
    Returns: ArgumentParser

    Raises: NULL

    """
    arg_parser = argparse.ArgumentParser(description="text2sql main program")
    arg_parser.add_argument("--config", required=True,
                                            help="config file for data reader, model, trainer and so on")
    arg_parser.add_argument("--device", default=None, type=str.lower,
                                            help="device: cpu|gpu. you can specific visible cuda device like gpu:0."
                                                 "if not setted, use the setting in json config file")
    arg_parser.add_argument("--checkpoint", help="only-for-train: pre-trained model checkpoint path.")
    arg_parser.add_argument("--parameters", help="only-for-train: pre-trained model parameters path.")
    arg_parser.add_argument("--infer-model", help="only-for-inference: saved inference model path.")
    arg_parser.add_argument("--save-path", help="in training process: pre-trained model parameters path. "
                                                "in predicting process: predict result file path.")
    arg_parser.add_argument("--data-path", help="train/dev/test data path root. replace <DATA_ROOT> in config file")
    arg_parser.add_argument("--db-max-len", type=int, nargs=3, default=(12, 40, 60),
                                            help="max len of db tables/columns/values. "
                                                 "replace <MAX_TABLE>, <MAX_COLUMN>, <MAX_VALUE> in config file")
    arg_parser.add_argument("--use-question-fea", default="yes", type=str.lower, help="yes|no")
    arg_parser.add_argument("--use-table-fea", default="yes", type=str.lower, help="yes|no")
    arg_parser.add_argument("--use-column-fea", default="yes", type=str.lower, help="yes|no")
    arg_parser.add_argument("--use-value-fea", default="yes", type=str.lower, help="yes|no")

    arg_parser.add_argument("--seed", help="random seed. currently unsupported!")
    arg_parser.add_argument("--log-file", dest="log_file",
                                            help="Log file path. Default is None, and logs will be wrote to stderr.")
    arg_parser.add_argument("--verbose", default=False, action="store_true",
                                            help="Runing in verbose mode, or not. Default is False. ")

    return arg_parser


def init_args():
    """init arguments from sys.argv
    Returns: parsed ArgumentParser

    Raises: NULL

    """
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    logging.info("commad line arguments: %s", str(args))

    g.use_question_feature = args.use_question_fea == 'yes'
    g.use_table_feature = args.use_table_fea == 'yes'
    g.use_column_feature = args.use_column_fea == 'yes'
    g.use_value_feature = args.use_value_fea == 'yes'

    if args.log_file is not None:
        log_dir = Path(args.log_file).parent
        if not log_dir.is_dir():
            log_dir.mkdir(parents=True)
    ## currently, it's not working correctly.
    ##if args.device.startswith('gpu:'):
    ##    visible = args.device.split(':', 1)[1]
    ##    os.environ['CUDA_VISIBLE_DEVICES'] = visible
    ##    logging.info('visible cuda devices is setted to %s', visible)

    return args


def _update_cuda_setting(device, worker_config):
    """udpate 'use_cuda' config in worker_config

    Args:
        device (str): cpu|gpu
        worker_config (dict): NULL

    Returns: None

    Raises: NULL
    """
    if device == 'cpu':
        worker_config['PADDLE_USE_GPU'] = 0
    elif device == 'gpu':
        worker_config['PADDLE_USE_GPU'] = 1
    else:
        pass   # use config_dict['trainer']['use_cuda'] as default


def _update_config(args, config_dict):
    """update param_dict by args. config in args is prior to config_dict.

    Args:
        args (ArgumentParser): [in] parsed aguemnt parser
        config_dict (dict): [in/out] config dict

    Returns: None

    Raises: NULL
    """
    if 'trainer' in config_dict:
        train_config = config_dict['trainer']
        _update_cuda_setting(args.device, train_config)

        if args.save_path is not None:
            train_config['output_path'] = args.save_path
        if args.checkpoint is not None:
            train_config['load_checkpoint'] = args.checkpoint
        if args.parameters is not None:
            train_config['load_parameters'] = args.parameters

    if 'predictor' in config_dict:
        predict_config = config_dict['predictor']
        _update_cuda_setting(args.device, predict_config)

        if args.infer_model is not None:
            predict_config['inference_model_path'] = args.infer_model
        if args.save_path is not None:
            predict_config['save_predict_file'] = args.save_path


def init_config(args, config_file, db_schema_max):
    """load config from file, and update db_schema_max

    Args:
        args (TYPE): NULL
        config_file (TYPE): NULL
        db_schema_max (TYPE): NULL

    Returns: TODO

    Raises: NULL
    """
    with open(args.config) as ifs:
        config_str = ifs.read()

    max_table, max_column, max_value = db_schema_max
    Grammar.MAX_TABLE = max_table
    Grammar.MAX_COLUMN = max_column
    Grammar.MAX_VALUE = max_value

    config_str = config_str.replace('<MAX_TABLE>', str(max_table))
    config_str = config_str.replace('<MAX_COLUMN>', str(max_column))
    config_str = config_str.replace('<MAX_VALUE>', str(max_value))
    if args.data_path is not None:
        config_str = config_str.replace('<DATA_ROOT>', args.data_path)
    logging.debug('config file:%s', config_str.rstrip())

    param_dict = json.loads(config_str)
    _update_config(args, param_dict)
    return param_dict


if __name__ == "__main__":
    """run some simple test cases"""
    args = init_args()
    for k, v in vars(args).items():
        print(k, v, sep=': ')

