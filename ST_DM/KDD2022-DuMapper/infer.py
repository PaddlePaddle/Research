#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: infer.py
func: 测试启动代码 
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/08/06
"""
import os
from tqdm import tqdm
import sys  

import numpy as np

from mmflib.utils import config
from mmflib.engine.infer import InferenceModel


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, show=False)
    infer = InferenceModel(config, 'l')
    #infer.unzip_pth()
    img_dir = "./sb_match_data2.4"
    path = "cfcc692d-5103-48ae-b676-e8624218f4ef_board.jpg"
    img_path = os.path.join(img_dir, path)
    x = "116.408971041"
    y = "39.9702910639"
    word = np.array([[float(x), float(y)]])
    fea = ' '.join([str(x) for x in infer(img_path, word)])
    print(fea)