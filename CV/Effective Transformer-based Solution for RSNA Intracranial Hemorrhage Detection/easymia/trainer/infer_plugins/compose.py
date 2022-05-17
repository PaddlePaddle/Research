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

import paddle
from easymia.libs import manager

@manager.PLUGINS.add_component
class ComposePlugins(object):
    """
    TBD
    """
    def __init__(self, plugins):
        self.plugins = plugins
        self.input_bs = None
        self.enable = False

        if len(self.plugins) > 0: self.enable = True

    def pre(self, samples, working_mode):
        """
        TBD
        """
        self.input_bs = samples.shape[0]
        for plugin in self.plugins:
            if working_mode == "eval" and plugin.only_infer:
                continue
            samples = plugin.pre(samples)
        return samples

    def __call__(self, model, samples, sample_ids, working_mode="eval"):
        """
        TBD, 这里有问题，会导致推理非常慢
        """
        samples = self.pre(samples, working_mode)
        sample_bs = samples.shape[0]

        # for sliding window and other TTA
        if sample_bs > self.input_bs:
            splitation = [self.input_bs] *\
                     (samples.shape[0] // self.input_bs) + [-1] * (samples.shape[0] % self.input_bs > 0)
            
            samples = paddle.split(samples, splitation, axis=0)
            for sample in samples:
                logits = model(sample) # list or dict
                logits = self.step(logits, working_mode)
        else:
            logits = model(samples)
            logits = self.step(logits, working_mode)

        return self.post(logits, sample_ids, working_mode)

    def step(self, logits, working_mode):
        """
        TBD
        """
        for plugin in self.plugins[::-1]:
            if working_mode == "eval" and plugin.only_infer:
                continue
            logits = plugin.step(logits)
        return logits

    def post(self, logits, sample_ids, working_mode):
        """
        TBD
        """
        for plugin in self.plugins[::-1]:
            if working_mode == "eval" and plugin.only_infer:
                continue
            logits = plugin.post(logits, sample_ids)
        return logits

    def dump(self, logits, sample_ids, working_mode):
        """
        TBD
        """
        for plugin in self.plugins[::-1]:
            if working_mode == "eval" and plugin.only_infer:
                continue
            logits = plugin.dump(logits, sample_ids)
        return logits