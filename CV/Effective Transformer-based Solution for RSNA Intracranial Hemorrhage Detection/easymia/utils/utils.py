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
训练器
"""

import os
import random
import time

import numpy as np
import paddle

def worker_init_fn(worker_id):
    """
    dataloader worker的初始函数
    根据worker_id设置不同的随机数种子，避免多个worker输出相同的数据
    """
    np.random.seed(random.randint(0, 100000))


def export_inference_model(model, 
                           input_shape, 
                           batch_size=0, 
                           input_dtype="float32",
                           save_dir="./",
                           save_prefix='model'):
    """
    Inference Model导出工具
    """
    assert isinstance(model, paddle.nn.Layer)
    assert isinstance(input_shape, (tuple, list))
    assert batch_size >= 0
    
    if model.mode == "clas":
        model.forward = model.__clas__
    elif model.mode == "det":
        model.forward = model.__det__
    elif model.mode == "seg":
        model.forward = model.__seg__
    elif model.mode == "pretrain":
        model.forward = model.__pretrain__

    input_shape = [None] + input_shape if batch_size == 0 else [batch_size] + input_shape

    model.eval()

    static_model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
            shape=input_shape, dtype=input_dtype)
    ])

    # 动转静模型存储接口
    paddle.jit.save(static_model, os.path.join(save_dir, save_prefix))
    print("Save Inference Model successfully, check `{}`".format(save_dir))


def dict_depth(d):
    """
    递归地获取一个dict的深度

    d = {'a':1, 'b': {'c':{}}} --> depth(d) == 3
    """
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0


def get_strategy():
    """
    TBD
    """
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.fuse_bn_act_ops = True
    build_strategy.enable_auto_fusion = True
    build_strategy.fuse_relu_depthwise_conv = True
    build_strategy.fuse_broadcast_ops = True
    build_strategy.fuse_all_optimizer_ops = True
    build_strategy.enable_inplace = True

    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.build_strategy = build_strategy
    return strategy


def drop_overtime_files(folder, keepSec=7200):
    """
    用来删除某个文件夹下 keepSec 秒之前创建的文件
    比如/dev/shm
    """
    now = time.time()

    for f in os.listdir(folder):
        try:
            create_t = os.path.getctime(os.path.join(folder, f))
            if now - create_t > keepSec:
                os.remove(os.path.join(folder, f))
        except:
            pass