# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as FL
from paddle.fluid.param_attr import ParamAttr

from . import resnet
from . import lesionnet
from . import densenet
from . import inception_v4 as inception

from learning_rate import cosine_decay_with_warmup

__all__ = ["create_model"]

ResNetModels = sorted(name for name in resnet.__dict__
                             if callable(resnet.__dict__[name]))
DenseNetModels = sorted(name for name in densenet.__dict__
                             if callable(densenet.__dict__[name]))

class create_model():
    def __init__(self, data_shape, loss_type, main_arch, name, model_type='train', **kwargs):
        """初始化函数"""
        self.loss_type = loss_type
        self.data_shape = data_shape
        self.main_arch = main_arch
        self.name = name
        self.model_type = model_type

    def create_acc_op(self, predict, label):
        """compute accuracy with tensor

         Args:
         predict: model output tensor activated by softmax
         label: a non-sparse tensor

         Returns:
         acc: acc tensor
         """
        accuracy = FL.accuracy(input=predict, label=label)
        return accuracy

    def create_reader_op(self, feed_list):
        """create reader with op

         Args:
         feed_list: placeholders needs to be feed

         Returns:
         reader: a reader
         """
        reader = fluid.io.DataLoader.from_generator(feed_list=feed_list, 
                     capacity=256, iterable=True, use_double_buffer=True)

        return reader

    def create_loss_op(self, predict, label, epsilon=1e-7):
        """compute loss with tensor

         Args:
         predict: model output tensor activated by softmax
         label: a non-sparse tensor

         Returns:
         loss: cross-entropy loss
         """
        if self.loss_type == "nl" and self.model_type == "train":
            one_hot_label = fluid.one_hot(label, depth=predict.shape[-1])
            one_hot_label = FL.squeeze(one_hot_label, axes=[-2])
            # log
            neg_prob = 1 - predict
            log_neg_prob = FL.log(fluid.layers.clip(neg_prob, min=epsilon, max=1.))
            ce_loss = -1 * log_neg_prob * one_hot_label
            cost = FL.reduce_sum(ce_loss, dim=-1, keep_dim=True)
        else: # PL or evaluation
            cost = FL.cross_entropy(predict, label)

        loss = FL.mean(cost)

        return loss

    def create_cam_op(self, predict, class_dim, heatmaps):
        """compute loss with tensor

         Args:
         predict: model output tensor activated by softmax
         class_dim: dim of multi-class vector
         heatmaps: 全局池化前的特征图

         Returns:
         heatmaps: class activation map
         """
        if self.main_arch in DenseNetModels:
            weights_shape = 1024
            name = "fc_weights"
        elif self.main_arch == "xception":
            weights_shape = 2048
            name = "fc_weights"
        else:
            raise ValueError("Calc CAM of model arch {} is not supported.".format(self.main_arch))

        fc_weights = FL.create_parameter(shape=[weights_shape, class_dim], dtype='float32', name=name) # 1024, 5

        pred_idx = FL.argmax(predict, 1) # bs, 1
        fc_weights = FL.transpose(fc_weights, perm=[1, 0]) # 5, 1024
        fc_weights = FL.gather(fc_weights, index=pred_idx) # bs, 1024

        heatmaps = heatmaps * fc_weights # bs, 1024, 16, 16
        heatmaps = FL.reduce_sum(heatmaps, dim=1, keep_dim=False)

        return heatmaps

    def net(self, class_dim=5, CAM=False):
        """Create second stage model
         Args:
         class_dim: dim of multi-class vector
         CAM:  是否创建CAM heatmap
         Returns:
         * A list contain 4/5 tensors / ops:
             - loss, cross-entropy loss tensor
             - accuracy, accuracy metric tensor
             - predict, model output tensor activated by softmax
             - hacked_img_id, img_id tensor
             - cam_heatmap, only if CAM == True, class activation map tensor
         * reader, reader op to feed data into placeholder
         """
        self.input_feature = fluid.data(name='{}_input'.format(self.name), shape=[-1] + self.data_shape, dtype='uint8')
        self.label = fluid.data(name='{}_label'.format(self.name), shape=[-1, 1], dtype='int64')
        self.img_id = fluid.data(name='{}_img_id'.format(self.name), shape=[-1, 1], dtype='int64')

        # Lesion Net
        lesion = lesionnet.LesionNet()

        # Backbone
        if self.main_arch in ResNetModels:
            model = resnet.__dict__[self.main_arch]()
        elif self.main_arch in DenseNetModels:
            model = densenet.__dict__[self.main_arch]()
        elif self.main_arch == "inception":
            model = inception.InceptionV4()
        else:
            raise ValueError("Model {} is not supported.".format(self.main_arch))

        inp = FL.transpose(FL.cast(self.input_feature, "float32"), perm=[0, 3, 1, 2]) / 255.

        # Element wise mul of lesion prob maps and input image
        lesion_probs = lesion.net(inp, class_dim=4) # bs, 4, 16, 16
        lesion_probs = FL.split(lesion_probs, num_or_sections=4, dim=1) # probs, bs*1*16*16 4

        I = FL.image_resize(inp, out_shape=(512, 512), resample="BILINEAR")
        Is = []
        
        for L in lesion_probs:
            W = FL.image_resize(L, out_shape=(512, 512), resample="NEAREST") # bs, 1, 512, 512
            temp_I = FL.elementwise_mul(I, FL.expand(W + 1., expand_times=[1, 3, 1, 1])) # W + 1., bs, 3, 512, 512
            Is.append(temp_I)
        I = FL.concat(Is, axis=1) # bs, 3*4, 512, 512
        I.stop_gradient = True
        
        lesion_pos_prob = 1. - lesion_probs[0]
        main_arch_out = model.net(I, class_dim=class_dim, lesion_map=lesion_pos_prob, CAM=CAM)
        
        if CAM:
            logit, heatmaps = main_arch_out
        else:
            logit = main_arch_out
        
        predict = FL.softmax(logit)
        accuracy = self.create_acc_op(predict, self.label)
        loss = self.create_loss_op(predict, self.label)
        reader = self.create_reader_op([self.img_id, self.input_feature, self.label])
        
        # This is a hack
        hacked_img_id = FL.cast(self.img_id, "int32")
        
        if CAM: 
            cam_heatmap = self.create_cam_op(predict, class_dim, heatmaps)
            return [loss, accuracy, predict, hacked_img_id, cam_heatmap], reader
        
        return [loss, accuracy, predict, hacked_img_id], reader