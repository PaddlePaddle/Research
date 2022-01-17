#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: train.py
func: 训练启动代码 
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/19
"""
import os
import sys

import paddle
import numpy as np

from mmflib.utils import config
from mmflib.engine.trainer import Trainer


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, show=False)

    trainer = Trainer(config, mode="train")
    trainer.train()