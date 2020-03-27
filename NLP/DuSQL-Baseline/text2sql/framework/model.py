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
:py:class:`Model` is an abstract class representing
"""


class Model(object):
    """Model:建模过程主要通过前向计算组网、构建优化器和指定指标评估方式来实现。
    """

    def __init__(self, model_params):
        """"""
        self.model_params = model_params 

    def forward(self, fields_dict, phase):
        """
        必须选项，否则会抛出异常。
        核心内容是模型的前向计算组网部分，包括loss值的计算，必须由子类实现。输出即为对输入数据执行变换计算后的结果。
        :param: fields_dict
                {"field_name":
                    {"RECORD_ID":
                        {"SRC_IDS": [ids],
                         "MASK_IDS": [ids],
                         "SEQ_LENS": [ids]
                        }
                    }
                }
        序列化好的id，供网络计算使用。
        :param: phase: 当前调用的阶段，包含训练、评估和预测，不同的阶段组网可以不一样。
                训练：TRAINING
                评估：EVALUATE
                预测：SAVE_INFERENCE
                
        :return: 训练：dict
                     {
                        "PREDICT_RESULT": [predictions],
                        "LABEL": [label],
                        "LOSS": [avg_cost]
                     }
                 预测：dict
                     {
                        "TARGET_FEED_NAMES": [ids, id_lens],
                        "TARGET_PREDICTS": [predictions]
                     }      
        实例化的dict，存放TARGET_FEED_NAMES, TARGET_PREDICTS, PREDICT_RESULT,LABEL,LOSS等希望从前向网络中获取的数据。
        """
        raise NotImplementedError

    def fields_process(self, fields_dict, phase):
        """
        对fields_dict中序列化好的id按需做二次处理。
        :return: 处理好的fields
        """
        raise NotImplementedError

    def make_embedding(self, fields_dict, phase):
        """构造embedding
        :param fields_dict:
        :param phase:
        :return: embedding_dict
        """
        raise NotImplementedError

    def optimizer(self, loss, is_fleet=False):
        """
        必须选项，否则会抛出异常。
        设置优化器，如Adam，Adagrad，SGD等。
        :param loss:前向计算得到的损失值。
        :param is_fleet:是否为分布式训练。
        :return:OrderedDict: 该dict中存放的是需要在运行过程中fetch出来的tensor，大多数情况下为空，可以按需求添加内容。
        """
        raise NotImplementedError

    def parse_predict_result(self, predict_result):
        """按需解析模型预测出来的结果
        :param predict_result: 模型预测出来的结果
        :return:None
        """
        raise NotImplementedError

    def get_metrics(self, fetch_output_dict, meta_info, phase):
        """指标评估部分的动态计算和打印
        :param fetch_output_dict: executor.run过程中fetch出来的forward中定义的tensor
        :param meta_info：常用的meta信息，如step, used_time, gpu_id等
        :param phase: 当前调用的阶段，包含训练和评估
        :return:metrics_return_dict：该dict中存放的是各个指标的结果，以文本分类为例，该dict内容如下所示：
                 {
                         "acc": acc,
                         "precision": precision
                }
        """

        raise NotImplementedError

    def metrics_show(self, result_evaluate):
        """评估指标展示
        :return:
        """
        raise NotImplementedError
