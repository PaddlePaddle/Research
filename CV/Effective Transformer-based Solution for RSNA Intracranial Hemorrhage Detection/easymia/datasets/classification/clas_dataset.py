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
分类数据集基类
"""

import os

import paddle
import numpy as np
from PIL import Image

from easymia.core.abstract_dataset import Dataset
from easymia.libs import manager
from easymia.transforms import Compose

@manager.DATASETS.add_component
class ClasDataset(Dataset):
    """
    分类数据集基类
    """
    def __init__(self, split='train', transforms=None, dataset_root=None):
        """
        Init
        """
        super().__init__(split)
        self.transforms = Compose('clas', transforms) \
                            if isinstance(transforms, (list, tuple)) else transforms
        self.dataset_root = dataset_root
