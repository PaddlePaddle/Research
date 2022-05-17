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
Training schedules
"""

from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer.lr import CosineAnnealingDecay

from easymia.libs import manager

@manager.SCHEDULES.add_component
class WarmupCosine(LinearWarmup):
    """
    Cosine learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): cosine decay

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self, warmed_lr, warmup_steps, decay_steps, **kwargs):
        start_lr = 0.0
        lr_sch = CosineAnnealingDecay(warmed_lr, decay_steps)

        super(WarmupCosine, self).__init__(
            learning_rate=lr_sch,
            warmup_steps=warmup_steps,
            start_lr=start_lr,
            end_lr=warmed_lr)

        self.update_specified = False