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
基于脑CT进行脑部出血分类
"""

import os

import paddle
import numpy as np

from easymia.tasks.task import Task
from easymia.libs import manager
from easymia.transforms import Compose

@manager.TASKS.add_component
class BranIHClasTask(Task):
    """
    RSNA-MICCAI Brain Intracranial Hemorrhage Classification (BrainIHClasTask)

    Args:
        datasets: a list of datasets
        transforms: data transforms in easymia framework 
    
    Task Usage:
        Previously,the datasets are separated from Tasks 
        Firstly,you can pick up a set of datasets from our easymia.datasets
        Then,you would compose them as a list to feed to Task
    """
    def __init__(self, datasets, transforms=None, mode='train'):
        super().__init__(datasets)
        self.transforms = Compose('clas', transforms) \
                            if isinstance(transforms, (list, tuple)) else transforms

    def collate_fn(self):
        """
        default paddle.fluid.dataloader.collate.default_collate_fn
        """
        return self.datasets[0].collate_fn()

    def __getitem__(self, idx):
        """
        getitem from datasets
        """
        dataset_id, dataset_idx = self.file_dataset_mapping[idx]
        sample = self.datasets[dataset_id].__getitem__(dataset_idx) # img, ..., sample_id
        if self.transforms:
            sample = list(sample)
            img = sample[0]
            img = self.transforms(img.transpose(1, 2, 0))
            sample[0] = img.transpose(2, 0, 1)

        return sample
