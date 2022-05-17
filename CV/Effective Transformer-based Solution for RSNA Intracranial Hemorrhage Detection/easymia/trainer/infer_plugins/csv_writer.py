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
模型输出结果->csv
"""

from datetime import datetime

import paddle.nn.functional as F

from easymia.libs import manager

@manager.PLUGINS.add_component
class CSVWritePlugin(object):
    """
    模型输出结果->csv
    """
    def __init__(self, use_argmax=False, argmax_dim=-1, logit_idx=0, save_path=None, only_infer=True):
        """
        use_argmax: bool,    是否需要执行argmax操作
        argmax_dim: int,     对哪个维度执行argmax操作
        logit_idx:  int,     模型输出为一个list，要记录list中第几个元素的输出
        save_path:  None|str,csv记录的路径
        """
        self.use_argmax = use_argmax
        self.argmax_dim = argmax_dim
        self.logit_idx = logit_idx
        self.only_infer = only_infer

        if save_path is None:
            self.save_path = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + "_csvwriter.csv"
        else:
            self.save_path = save_path

        self.file_handle = None

    def pre(self, samples):
        """
        TBD
        """
        return samples

    def step(self, logits):
        """
        TBD
        """
        return logits

    def post(self, logits, sample_ids):
        """
        TBD
        """
        return logits

    def dump(self, logits, sample_ids):
        """
        step
        logits: list[paddle tensor], shape with [N, C, d1, d2, ...]
        labels: paddle tensor, shape with [N, d1, d2, ...]
        """
        if self.file_handle is None:
            self.file_handle = open(self.save_path, "w+")

        logits = F.sigmoid(logits[self.logit_idx]).numpy()
        
        sample_ids = sample_ids.numpy().tolist()

        if self.use_argmax:
            logits = logits.argmax(self.argmax_dim)

        for i, j in zip(logits, sample_ids):
            msg = "{},{}\n".format(j, " ".join([str(item) for item in i.tolist()]))
            self.file_handle.write(msg)
