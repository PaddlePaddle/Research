#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: train.py
Author: fanmiao@baidu.com
Date: 2022/01/09
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
