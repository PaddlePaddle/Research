# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
训练器
"""

import os
import time
import shutil
from collections import deque

import paddle
import numpy as np

from easymia.utils import utils
from easymia.utils import logger
from easymia.utils import progbar
from .evaluator import Evaluator

import warnings
warnings.filterwarnings("ignore")

class Trainer(Evaluator):
    """
    TBD
    """
    def __init__(self, 
                 model, 
                 optimizer, 
                 criterion,
                 iters,
                 train_task_loader, 
                 val_task_loader, 
                 metrics, 
                 batch_size=[1, 1],
                 num_workers=0,
                 save_dir="./output/",
                 save_interval=100,
                 log_iters=10,
                 keep_checkpoint_max=5,
                 keep_checkpoint_iter=None,
                 overwrite_save_dir=False,
                 convert_syncbn=False,
                 infer_plugins=None,
                 multigpu_infer=False):
        """
        TBD
        """
        # super self.model, self.metrics, self.batch_size, self.num_workers, self.val_loader, self.val_samples
        if convert_syncbn:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        super().__init__(model, val_task_loader, metrics, infer_plugins, batch_size, num_workers, multigpu_infer)
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()

        if nranks > 1:
            paddle.distributed.fleet.init(is_collective=True, strategy=utils.get_strategy())
            self.optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)
            self.ddp_model = paddle.distributed.fleet.distributed_model(model)
        else:
            self.optimizer = optimizer
            self.ddp_model = None

        self.criterion = criterion
        self.iters = iters
        self.do_eval = val_task_loader is not None
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.log_iters = log_iters
        self.keep_checkpoint_max = keep_checkpoint_max
        self.keep_checkpoint_iter = keep_checkpoint_iter

        self.train_task_loader = train_task_loader
        self.recorder = logger.Recorder(
            ['loss', "reader_cost_time", "batch_cost_time", "lr"] + [m.name for m in metrics])

        self.train_loader, self.train_samples = self.create_loader(self.train_task_loader, self.batch_size[0], True)

        if os.path.exists(self.save_dir) and local_rank == 0:
            if overwrite_save_dir:
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)
            else:
                raise ValueError("Output dir {} exists, change output dir \
                    or set `overwrite_save_dir` == True".format(self.save_dir))

    def print_status(self, iter, reset=False):
        """
        print
        """
        avg_loss = self.recorder.get("loss", reduction="mean")

        avg_batch_cost = self.recorder.get("batch_cost_time", reduction="mean")
        avg_reader_cost = self.recorder.get("reader_cost_time", reduction="mean")
        lr = self.recorder.get("lr")[-1]

        logger.info(
            "[TRAIN] Iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}"
                    .format(iter, self.iters, avg_loss, lr, avg_batch_cost, avg_reader_cost))

        if reset:
            self.recorder.clear("loss")
            self.recorder.clear("batch_cost_time")
            self.recorder.clear("reader_cost_time")
            self.recorder.clear("lr")

    def loss_computation(self, outputs, labels, criterions, coefs):
        """
        loss计算
        """
        assert len(outputs) == len(criterions), \
            'The length of outputs should equal to the types of loss config: {} != {}.'\
                .format(len(outputs), len(criterions))

        if isinstance(labels, (list, tuple)):
            assert len(labels) == 1 or len(outputs) == len(labels), \
                'The length of outputs should equal to the labels: {} != {}.'.format(len(outputs), len(labels))
            if len(labels) == 1: labels = labels[0]

        loss_list = []
        for i in range(len(outputs)):
            if isinstance(labels, (list, tuple)):
                loss_list.append(coefs[i] * criterions[i](outputs[i], labels[i]))
            else:
                loss_list.append(coefs[i] * criterions[i](outputs[i], labels))
        return sum(loss_list)

    def train(self, start_iter, end_iter):
        """
        模型训练for-loop
        """
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()

        iterator = self.train_loader.__iter__()
        self.model.train()
        batch_start = time.time()

        iter = start_iter
        while iter < end_iter:
            try:
                data = iterator.next()
            except StopIteration:
                iterator = self.train_loader.__iter__()
                data = iterator.next()

            iter += 1
            if iter > end_iter: break

            self.recorder.record("reader_cost_time", time.time() - batch_start)
            
            images, labels, _ = self.unpack_data(data)

            if nranks > 1:
                outputs = self.ddp_model(images)
            else:
                outputs = self.model(images)

            if isinstance(outputs, dict):
                outputs, labels = self.flatten_dict_output(outputs, labels)

            loss = self.loss_computation(outputs, labels, self.criterion['types'], self.criterion['coef'])

            loss.backward()
            self.optimizer.step()
            self.model.clear_gradients()

            current_lr = self.optimizer.get_lr()
            # update lr
            if isinstance(self.optimizer, paddle.distributed.fleet.Fleet):
                lr_sche = self.optimizer.user_defined_optimizer._learning_rate
            else:
                lr_sche = self.optimizer._learning_rate

            if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                lr_sche.step()

            self.recorder.record("batch_cost_time", time.time() - batch_start)
            self.recorder.record("loss", float(loss))
            self.recorder.record("lr", current_lr)

            if iter % self.log_iters == 0 and local_rank == 0:
                self.print_status(iter, reset=True)

            batch_start = time.time()
        iterator.__del__()
        if local_rank == 0:
            utils.drop_overtime_files("/dev/shm/", keepSec=7200)

    def fit(self):
        """
        外部调用接口
        """
        local_rank = paddle.distributed.ParallelEnv().local_rank
        # epochs = self.iters // len(train_task_loader) + 1
        steps = self.iters // self.save_interval + 1
        best_benchmark = 0.
        best_model_iter = 0.
        save_models = deque()

        for step in range(steps):
            start_iter = step * self.save_interval
            end_iter = min((step + 1) * self.save_interval, self.iters)

            self.train(start_iter, end_iter)
            if self.do_eval:
                benchmark = self.evaluate()

            if (end_iter % self.save_interval == 0 or end_iter == self.iters) and local_rank == 0:
                current_save_dir = os.path.join(self.save_dir, "iter_{}".format(end_iter))
                if not os.path.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                paddle.save(self.model.state_dict(),
                            os.path.join(current_save_dir, 'model.pdparams'))
                paddle.save(self.optimizer.state_dict(),
                            os.path.join(current_save_dir, 'model.pdopt'))

                if not (self.keep_checkpoint_iter is not None and end_iter in self.keep_checkpoint_iter):
                    save_models.append(current_save_dir)

                if len(save_models) > self.keep_checkpoint_max > 0:
                    model_to_remove = save_models.popleft()
                    shutil.rmtree(model_to_remove)

                if self.do_eval:
                    if benchmark > best_benchmark:
                        best_benchmark = benchmark
                        best_model_iter = end_iter

                        best_model_dir = os.path.join(self.save_dir, "best_model")
                        paddle.save(self.model.state_dict(),
                                    os.path.join(best_model_dir, 'model.pdparams'))
                    logger.info(
                            '[EVAL] The model with the best validation benchmark ({:.4f}) was saved at iter {}.'
                            .format(best_benchmark, best_model_iter))

                for m in self.metrics:
                    m.clear()
