# coding: utf8
import sys
import paddle.fluid as fluid
import numpy as np
import importlib
from src.utils.config import cfg


def multi_cross_entropy_loss(pred1, pred2, label, num_classes=19, weight=None):
    # main_loss + aux_loss
    #pred1.shape:  nchw
    #pred2.shape:  nchw
     
    pred1 = fluid.layers.transpose(pred1, perm=[0, 2, 3, 1]) # n,h,w, c
    pred1 = fluid.layers.reshape(pred1, [-1, num_classes])# n*h*w, c

    pred2 = fluid.layers.transpose(pred2, perm=[0, 2, 3, 1])
    pred2 = fluid.layers.reshape(pred2, [-1, num_classes])
 
    label = fluid.layers.reshape(label, [-1, 1]) # n h w -> n*h*w, 1                                                                                                          
    pred1 = fluid.layers.softmax(pred1, use_cudnn=False, axis=1) # 
    loss1 = fluid.layers.cross_entropy(pred1, label, ignore_index=cfg.DATASET.IGNORE_INDEX)
     
    pred2 = fluid.layers.softmax(pred2, use_cudnn=False, axis=1)
    loss2 = fluid.layers.cross_entropy(pred2, label, ignore_index=cfg.DATASET.IGNORE_INDEX)
    label.stop_gradient = True
    total_loss=loss1*cfg.MODEL.MULTI_LOSS_WEIGHT[0]+loss2*cfg.MODEL.MULTI_LOSS_WEIGHT[1]
    if weight is not None:
        total_loss = total_loss * weight
        weight.stop_gradient = True
    return total_loss
