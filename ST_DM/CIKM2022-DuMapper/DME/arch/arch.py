#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: arch.py
func: 多模态数据融合pipleline
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/20
"""
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from mmflib.arch.im2vec.custom_resnet50 import ResNetCustom
from mmflib.arch.word2vec.geohash import GeoHash
from mmflib.arch.fusebody.concat import ConcatFuse
from mmflib.arch.fusebody.attention import AttentionFuse
from mmflib.arch.gears.neck import BottleNeck
from mmflib.arch.gears.metric_head import ArcMarginProduct


def build_w2v(config):
    """建立word2vec模型
    """
    name = config.pop("name")
    word2vec = eval(name)(**config)

    return word2vec


def build_fuse(config):
    """建立word2vec模型
    """
    name = config.pop("name")
    fuseop = eval(name)(**config)

    return fuseop


def build_gear(config):
    """建立neck head
    """
    name = config.pop("name")
    gearop = eval(name)(**config)

    return gearop


class MMFModel(nn.Layer):
    """MMFModel
    """
    def __init__(self, config, mode="train"):
        super(MMFModel, self).__init__()

        im2vec_config = config['Image2Vec']
        im2vec_name = im2vec_config.pop("name")
        if "pretrained" in im2vec_config:
            pretrained = im2vec_config.pop("pretrained")
        else:
            pretrained = None
        self.im2vec = eval(im2vec_name)(**im2vec_config)
        if pretrained is not None:
            print("found pretrain model: {}".format(pretrained))
            param_state_dict = paddle.load(path + ".pdparams")
            self.im2vec.set_dict(param_state_dict)

        if "Word2Vec" in config:
            self.word2vec = build_w2v(config["Word2Vec"])
        else:
            self.word2vec = None
        
        if "FuseBody" in config:
            self.fuse = build_fuse(config["FuseBody"])
        else:
            self.fuse = None
        
        if "Neck" in config:
            self.neck = build_gear(config["Neck"])
            
        else:
            self.neck = None
        
        if "Head" in config and mode == "train":
            self.head = build_gear(config["Head"])
        else:
            self.head = None
        
    def forward(self, img, word, label=None):
        """forward"""
        img = self.im2vec(img)
        if self.word2vec is not None:
            word = self.word2vec(word)
        if self.fuse is not None:
            x = self.fuse(img, word)
        else:
            x = img
        if self.neck is not None:
            x = self.neck(x)
        if self.head is not None:
            y, x_norm = self.head(x, label)
        else:
            y = None
            x_norm = paddle.sqrt(paddle.sum(paddle.square(x), axis=1, keepdim=True))
        return {"features": x, "logits": y, "norm": x_norm}