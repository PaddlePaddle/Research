# -*- coding: utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
:py:class:`TrainerConfig` is an abstract class representing
"""


class TrainerConfig(object):
    """TrainerConfig
    """
    def __init__(self):
        self.do_train = True  # 训练模型时设置为True
        self.do_dev = False  # 训练过程中需要进行验证集评估的时候设置为True
        self.do_test = False  # 训练过程中需要进行测试集评估的时候设置为True
        self.do_predict = False  # 如果需要直接做预测，请设置为True
        self.save_steps = 500  # 模型保存的间隔, 即训练多少个batch之后保存一次模型
        self.eval_step = 100  # 当do_val或者do_test设置为True时生效，表示训练间隔多少个batch之后开始评估和预测
        self.train_log_step = 10  # 间隔多少个batch时打印训练日志
        self.is_local = True
        self.use_cuda = False
        self.load_checkpoint = False
        self.load_parameters = False
        self.pretrain_model = False

        # 需要做预测或者热启动的时候请填写该参数，其对应的模型为我们在训练过程中存储到./output/checkpoints中的数据
        # self.init_checkpoint = "./output/checkpoints/checkpoints_step_500"
