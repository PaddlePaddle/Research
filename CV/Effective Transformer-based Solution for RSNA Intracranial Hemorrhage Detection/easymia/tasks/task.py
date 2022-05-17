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
任务集基类
"""

import os

import paddle
import numpy as np

class Task(paddle.io.Dataset):
    """
    任务基类
    """
    def __init__(self, datasets):
        """
        Init
        datasets: list(dataset object)
        """
        self.datasets = datasets
        self.gen_dataset_mapping()

    def __getitem__(self, idx):
        """
        getitem
        """
        # dataset_id, dataset_idx = self.file_dataset_mapping[idx]
        # img, label = self.datasets[dataset_id][dataset_idx]
        # return img, label
        pass

    def __len__(self):
        return sum([len(item) for item in self.datasets])

    def gen_dataset_mapping(self):
        """
        global index -> dataset index
        file_dataset_mapping: key = global index
                              value = (dataset id, dataset index)
        """
        file_dataset_mapping = {}
        for k, dataset in enumerate(self.datasets):
            pre_max = max(file_dataset_mapping.keys()) + 1 if len(file_dataset_mapping) > 0 else 0
            file_dataset_mapping.update(
                {i + pre_max:(k, i) for i in range(len(dataset))})

        self.file_dataset_mapping = file_dataset_mapping

    def collate_fn(self):
        """
        临时解决方案
        todo: 变更为读取哪个数据集的内容，就用哪个数据集的collate_fn，不知道能否实现
        如果不能实现，就要求外部传入参数，选择使用哪个dataset的collate_fn
        """
        return self.datasets[0].collate_fn

if __name__ == '__main__':
    class RandomDataset(paddle.io.Dataset):
        """
        test dataset
        """
        def __init__(self, num_samples):
            """
            doc
            """
            self.num_samples = num_samples

        def __getitem__(self, idx):
            """
            doc
            """
            return idx, self.num_samples

        def __len__(self):
            """
            doc
            """
            return self.num_samples

    dataset1 = RandomDataset(5)
    dataset2 = RandomDataset(10)
    dataset3 = RandomDataset(15)

    task = Task(datasets=[dataset1, dataset2, dataset3])

    print(len(task)) # 30
    print(task.file_dataset_mapping)
    print(task.__getitem__(3)) # (3, 5)
    print(task.__getitem__(4)) # (4, 5)
    print(task.__getitem__(5)) # (0, 10)
    print(task.__getitem__(15)) # (0, 15)