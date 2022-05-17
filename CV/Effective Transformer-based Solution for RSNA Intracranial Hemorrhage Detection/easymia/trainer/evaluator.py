# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
评估/推理器
"""

import os
import numbers
import time
import shutil

import paddle
import numpy as np

from easymia.trainer import infer_plugins as ip
from easymia.utils import utils
from easymia.utils import logger
from easymia.utils import progbar

class Evaluator(object):
    """
    TBD
    """
    def __init__(self, 
                 model, 
                 val_task_loader, 
                 metrics=None, 
                 infer_plugins=[],
                 batch_size=[1, 1],
                 num_workers=0,
                 multigpu_infer=False):
        """
        TBD
        """
        self.model = model
        self.metrics = metrics
        self.infer_plugins = ip.ComposePlugins(infer_plugins)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_task_loader = val_task_loader
        self.multigpu_infer = multigpu_infer

        if self.val_task_loader is not None:
            self.val_loader, self.val_samples = self.create_loader(self.val_task_loader, self.batch_size[1])
        else:
            self.val_loader, self.val_samples = None, None

    def create_loader(self, task_loader, batch_size=1, training=False):
        """
        python loader -> paddle loader with multi-process
        """
        if self.multigpu_infer or training:
            batch_sampler = paddle.io.DistributedBatchSampler(
                task_loader, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            batch_sampler = None

        loader = paddle.io.DataLoader(
            task_loader,
            batch_size=batch_size if batch_sampler is None else 1,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            return_list=True,
            worker_init_fn=utils.worker_init_fn,
            collate_fn=task_loader.collate_fn()
        )

        return loader, len(task_loader)

    def infer(self):
        """
        infer
        """
        return self.run_inference(False)

    def evaluate(self):
        """
        infer
        """
        return self.run_inference(True)

    def metrics_step(self, outputs, labels):
        """
        metrics step

        logits_list: list(Tensor), multiple model outputs
        labels: Tensor, fitting target
        """
        assert len(outputs) == len(self.metrics), \
            'The length of outputs should equal to the types of metric config: {} != {}.'\
                .format(len(outputs), len(self.metrics))

        assert len(outputs) == len(labels), \
            'The length of outputs should equal to labels: {} != {}.'.format(len(outputs), len(labels))

        for i, (opt, lab) in enumerate(zip(outputs, labels)):
            self.metrics[i].step(opt, lab)

    def run_inference(self, evaluate=False):
        """
        TBD
        """
        working_mode = "eval" if evaluate else "infer"
        self.model.eval()

        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()

        if nranks > 1 and self.multigpu_infer:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
                paddle.distributed.init_parallel_env()

        if local_rank == 0: 
            evaluated_sample_ids = []

        progbar_val = progbar.Progbar(target=len(self.val_loader), verbose=1, interval=10)

        with paddle.no_grad():
            for iter, data in enumerate(self.val_loader):
                images, labels, sample_ids = self.unpack_data(data)

                if self.multigpu_infer or local_rank == 0:
                    if self.infer_plugins.enable:
                        outputs = self.infer_plugins(self.model, images, sample_ids, working_mode)
                    else:
                        outputs = self.model(images)
                else:
                    outputs = None
                if isinstance(outputs, dict):
                    outputs, labels = self.flatten_dict_output(outputs, labels)

                if nranks > 1 and self.multigpu_infer:
                    sample_ids = self.gather_tensor(sample_ids)
                    outputs = [self.gather_tensor(output) for output in outputs]
                    if labels: # is not None
                        labels = [self.gather_tensor(label) for label in labels]

                if local_rank == 0:
                    if evaluate:
                        keep_idx = []
                        for k, sample_id in enumerate(sample_ids.numpy().tolist()):
                            if sample_id not in evaluated_sample_ids: 
                                evaluated_sample_ids.append(sample_id)
                                keep_idx.append(k)
                        outputs = [output[keep_idx] for output in outputs]
                        sample_ids = sample_ids[keep_idx]
                        labels = [label[keep_idx] for label in labels]

                        self.metrics_step(outputs, labels) # 暂存计算metrics需要的数据

                    self.infer_plugins.dump(outputs, sample_ids, working_mode)

                    progbar_val.update(iter + 1)

        if local_rank == 0 and evaluate:
            msg = "\n[EVAL] #Images: {} ".format(len(evaluated_sample_ids))

            for metric_obj in self.metrics:
                metric_value = metric_obj.calc()
                if isinstance(metric_value, numbers.Number):
                    msg += "{}: {:.4f}  ".format(metric_obj.name, metric_value)
                else:
                    msg += "{}: {}  ".format(metric_obj.name, metric_value)
            logger.info(msg)

            return self.metrics[0].benchmark
        else:
            return 0

    def flatten_dict_output(self, output, label=None):
        """
        将字典类型的output与label展开为list，并确保其一一对应
        output: dict {key1: [tensor1, tensor2], key2: [tensor3, tensor4, ...]}
                        |                         |
                        v                         v
        label:  dict {key1: label1,             key2: label2}
        """
        if label is None:
            return [y for x in output.values() for y in x]

        assert isinstance(output, dict) and isinstance(label, dict), \
            "Model output and Label MUST be `dict`, got {} and {}.".format(type(output), type(label))

        assert all([k in label.keys() for k in output.keys()]), \
            "All keys in output must contained in label, got {}, {}".format(list(output.keys()), list(label.keys()))

        output_collector = []
        label_collector = []

        for key in output.keys():
            opt = output[key]
            lab = label[key]

            output_collector.extend(opt)
            label_collector.extend([lab] * len(opt))

        return output_collector, label_collector

    def gather_tensor(self, tensor, stack_axis=0):
        """
        多卡Tensor聚合
        """
        tensor_list = []
        paddle.distributed.all_gather(tensor_list, tensor)
        return paddle.concat(tensor_list, axis=stack_axis)

    def unpack_data(self, data):
        """
        数据解包
        """
        if isinstance(data, dict):
            images, labels, sample_ids = data.get("data"), data.get("label", None), data.get("index")
        elif isinstance(data, (list, tuple)):
            if len(data) == 2:
                images, sample_ids = data
                labels = None
            elif len(data) == 3:
                images, labels, sample_ids = data

        if images.dtype == paddle.uint8:
            images = (images / 255.).astype("float32")

        if isinstance(labels, paddle.Tensor):
            labels = [labels]

        return images, labels, sample_ids
