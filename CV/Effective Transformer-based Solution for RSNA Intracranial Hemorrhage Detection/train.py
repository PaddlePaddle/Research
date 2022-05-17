# -*-coding utf-8 -*-
##########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
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
#
##########################################################################

import argparse
import random

import paddle
import numpy as np

from easymia.libs import manager, Config
from easymia.utils import get_sys_env, logger
from easymia.trainer import Trainer


def parse_args():
    """
    command args
    """
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters for training',
        type=int,
        default=None)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=None)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save',
        type=int,
        default=5)
    parser.add_argument(
        '--keep_checkpoint_iter',
        dest='keep_checkpoint_iter',
        nargs='+',
        type=int,
        default=None)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Eval while training',
        action='store_true')
    parser.add_argument(
        '--convert_syncbn',
        dest='convert_syncbn',
        help='convert sync batch norm',
        action='store_true')
    parser.add_argument(
        '--overwrite_save_dir',
        dest='overwrite_save_dir',
        help='Overwrite the save dir if the path already exists.',
        action='store_true')
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every log_iters',
        default=10,
        type=int)
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
        default=None,
        type=int)
    parser.add_argument(
        '--multigpu_infer',
        dest='multigpu_infer',
        action='store_true')


    return parser.parse_args()


def main(args):
    """
    main
    """
    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)
    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size,
        resume_model=args.resume_model)

    train_dataset = cfg.train_dataset
    if train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )
    val_dataset = cfg.val_dataset if args.do_eval else None

    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)

    trainer = Trainer(model=cfg.model, 
                      optimizer=cfg.optimizer, 
                      criterion=cfg.loss, 
                      iters=cfg.iters,
                      train_task_loader=train_dataset, 
                      val_task_loader=val_dataset, 
                      metrics=cfg.metrics, 
                      batch_size=cfg.batch_size,
                      num_workers=args.num_workers,
                      save_dir=args.save_dir,
                      save_interval=args.save_interval,
                      log_iters=args.log_iters,
                      keep_checkpoint_max=args.keep_checkpoint_max,
                      keep_checkpoint_iter=args.keep_checkpoint_iter,
                      overwrite_save_dir=args.overwrite_save_dir,
                      convert_syncbn=args.convert_syncbn,
                      infer_plugins=cfg.infer_plugin,
                      multigpu_infer=args.multigpu_infer)

    trainer.fit()


if __name__ == '__main__':
    args = parse_args()
    main(args)