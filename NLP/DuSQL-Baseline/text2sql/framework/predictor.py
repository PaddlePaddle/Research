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
:predictor
"""
import logging
import time
import json
from paddle.fluid.core_avx import AnalysisConfig, create_paddle_predictor, PaddleTensor

from text2sql.framework.rule import InstanceName
from text2sql.framework.utils.util_helper import array2tensor


class Predictor(object):
    """Predictor: 模型预测
    """
    def __init__(self, param, data_set_reader, model_class):
        """
        1.解析input_data的结构 2.解析参数，构造predictor  3. 启动data_generator,开始预测 4.回掉预测结果到model中进行解析
        :param param: 运行的基本参数设置
        :param data_set_reader: 运行的基本参数设置
        :param model_class: 使用的是哪个model
        """
        self.data_set_reader = data_set_reader
        self.param = param
        self.model_class = model_class
        self.predictor = None
        self.input_keys = []
        self.init_data_params()
        self.init_env()

    def init_env(self):
        """
        :return:
        """
        model_path = self.param["inference_model_path"]
        config = AnalysisConfig(model_path + "/" + "model", model_path + "/" + "params")
        if self.param["PADDLE_USE_GPU"]:
            config.enable_use_gpu(1024)
        else:
            config.disable_gpu()
            config.enable_mkldnn()
        self.predictor = create_paddle_predictor(config.to_native_config())

    def init_data_params(self):
        """
        :return:
        """
        model_path = self.param["inference_model_path"]
        data_params_path = model_path + "/infer_data_params.json"
        with open(data_params_path) as ifs:
            param_dict = json.load(ifs)
        self.input_keys = param_dict.get("fields")

    def do_predict(self):
        """
        :return:
        """
        logging.debug("start do predict....")
        total_time = 0
        reader = self.data_set_reader.predict_reader.data_generator()

        for sample in reader():
            sample_dict = self.data_set_reader.predict_reader.convert_fields_to_dict(sample, need_emb=False)
            input_list = []
            for item in self.input_keys:
                kv = item.split("#")
                name = kv[0]
                key = kv[1]
                item_instance = sample_dict[name]
                input_item = item_instance[InstanceName.RECORD_ID][key]
                input_list.append(input_item)

            inputs = [array2tensor(ndarray) for ndarray in input_list]
            begin_time = time.time()
            result = self.predictor.run(inputs)
            end_time = time.time()
            total_time += end_time - begin_time
            self.model_class.parse_predict_result(result)

        logging.debug("total_time:{}".format(total_time))

