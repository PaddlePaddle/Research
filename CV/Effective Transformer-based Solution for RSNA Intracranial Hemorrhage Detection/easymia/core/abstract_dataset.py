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
import paddle

class Dataset(paddle.io.Dataset):
    """
    数据集基类
    """
    def __init__(self, split):
        """
        Init
        """
        split = split.lower()
        assert split in ['train', 'val', 'test', 'pretrain'], \
            "Arg split in config file must be `train`, `val`, `test` or `pretrain`, but got {}".format(split)
        self.split = split

    @staticmethod
    def collate_fn(batch):
        """
        default paddle.fluid.dataloader.collate.default_collate_fn
        """
        return paddle.fluid.dataloader.collate.default_collate_fn(batch)