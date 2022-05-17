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
"""
ACC metric
"""

import numpy as np
import paddle
import paddle.nn.functional as F
from easymia.libs import manager

@manager.METRICS.add_component
class NLLMetric(object):
    """
    Negative Log Loss for metric
    return 1 - nll(prob, label)
    """
    def __init__(self, weight=None, ignore_index=255, class_reduction='mean'):
        """
        Init
        """
        self.name = "nll"
        self.y1 = []
        self.y2 = []
        self.weight = weight
        self.ignore_index = ignore_index
        self.class_reduction = class_reduction

    def step(self, logits, labels):
        """
        step
        logits: paddle tensor, shape with [N, C, d1, d2, ...]
        labels: paddle tensor, shape with [N, C, d1, d2, ...]
        """
        assert logits.shape == labels.shape
        spatial_dims = list(range(2, logits.ndim))
        C = logits.shape[1]

        labels = labels.transpose([0] + spatial_dims + [1]).reshape([-1, C]).numpy().tolist()
        logits = logits.transpose([0] + spatial_dims + [1]).reshape([-1, C]).numpy().tolist()

        self.y1.extend(labels)
        self.y2.extend(logits)

    def clear(self):
        """
        clear
        """
        self.y1 = []
        self.y2 = []

    def calc(self):
        """
        calc
        """
        labels = np.array(self.y1, dtype='float32')
        pred = np.array(self.y2, dtype='float32')
        C = pred.shape[1]

        log_loss = []
        for c in range(C):
            keep = labels[:, c] != self.ignore_index
            log_loss.append(
                F.binary_cross_entropy_with_logits(
                    paddle.to_tensor(pred[keep, c]), 
                    paddle.to_tensor(labels[keep, c]), 
                    reduction="mean").numpy()
            )
        log_loss = np.array(log_loss).reshape(C)

        if self.weight is not None:
            weight = np.array(self.weight, dtype='float32')
            log_loss = log_loss * weight

        if self.class_reduction == "mean":
            reducted_log_loss = log_loss.mean()
        elif self.class_reduction == "sum":
            reducted_log_loss = log_loss.sum()

        ret_info = "mLogLoss={:.4f}, class LogLoss=[".format(reducted_log_loss) +\
                     " ".join(["{:.4f}".format(i) for i in log_loss]) + "]"
        return ret_info

    @property
    def benchmark(self):
        """
        benchmark
        """
        labels = np.array(self.y1, dtype='float32')
        pred = np.array(self.y2, dtype='float32')
        C = pred.shape[1]

        log_loss = []
        for c in range(C):
            keep = labels[:, c] != self.ignore_index
            log_loss.append(
                F.binary_cross_entropy_with_logits(
                    paddle.to_tensor(pred[keep, c]), 
                    paddle.to_tensor(labels[keep, c]), 
                    reduction="mean").numpy()
            )
        log_loss = np.array(log_loss).reshape(C)

        if self.weight is not None:
            weight = np.array(self.weight, dtype='float32')
            log_loss = log_loss * weight

        if self.class_reduction == "mean":
            reducted_log_loss = log_loss.mean()
        elif self.class_reduction == "sum":
            reducted_log_loss = log_loss.sum()

        return 1 - reducted_log_loss