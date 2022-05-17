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
交叉熵
"""

import paddle
import paddle.nn.functional as F

from easymia.core.abstract_loss import AbstractLoss
from easymia.libs import manager

@manager.LOSSES.add_component
class CrossEntropyLoss(AbstractLoss):
    """
    Implements the cross entropy loss function.
    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self, mode):
        super(CrossEntropyLoss, self).__init__(mode)

    def __clas__(self, logit, label, info=None):
        """
        Forward computation for clas
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1.
            info   (Dict):  Auxiliary information for loss calc. Not used in this func.
        """
        cost = F.cross_entropy(input=logit, label=label)
        avg_cost = paddle.mean(cost)
        return avg_cost
    
    def __seg__(self, logit, label, info=None):
        """
        Forward computation for seg
        Args:
            logit (Tensor): Logit tensor, the data type is int64. 
                            Shape is (N, H, W)
            label (Tensor): Label tensor, the data type is int64. 
                            Shape is (N, C, H, W), where C is number of classes
            info   (Dict):  Auxiliary information for loss calc. Not used in this func.
        """
        dims = [0] + list(range(2, logit.ndimension())) + [1]
        logit = paddle.transpose(logit, dims)
        label = label.astype("int64")
        cost = F.cross_entropy(input=logit, label=label)
        avg_cost = paddle.mean(cost)
        return avg_cost 