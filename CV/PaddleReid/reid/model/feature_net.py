from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


import os
import pdb

class feature_net():
    def __init__(self, act='leaky'):
        self.act = act

    def net(self, input, is_train = False, num_features = 256, class_dim = 751, finetune=False):
        reid_bn1 = fluid.layers.batch_norm(input=input,
                                  is_test = not is_train,
                                  name = 'reid_bn1',
                                  param_attr = fluid.ParamAttr(name='reid_bn1_scale'),
                                  bias_attr = fluid.ParamAttr(name='reid_bn1_offset'),
                                  moving_mean_name='reid_bn1_moving_mean',
                                  moving_variance_name='reid_bn1_moving_variance')
        if self.act == 'leaky':
            reid_bn1 = fluid.layers.leaky_relu(x=reid_bn1, alpha=0.1)


        stdv = 1.0 / math.sqrt(reid_bn1.shape[1]*1.0)
        reid_fc1 = fluid.layers.fc(input = reid_bn1,
                                    size = num_features,
                                    name = 'reid_fc1',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))

        reid_bn2 = fluid.layers.batch_norm(input=reid_fc1,
                                  is_test = not is_train,
                                  name = 'reid_bn2',
                                  param_attr = fluid.ParamAttr(name='reid_bn2_scale'),
                                  bias_attr = fluid.ParamAttr(name='reid_bn2_offset'),
                                  moving_mean_name='reid_bn2_moving_mean',
                                  moving_variance_name='reid_bn2_moving_variance',)


        stdv = 1.0 / math.sqrt(reid_bn2.shape[1]*1.0)
        if not finetune:
            reid_cls = fluid.layers.fc(input = reid_bn2,
                                    size = class_dim,
                                    name = 'reid_cls',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))
        else:                        
            reid_cls = fluid.layers.fc(input = reid_bn2,
                                    size = class_dim,
                                    name = 'reid_cls',
                                    param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv), learning_rate=10.0))

        return reid_cls, reid_bn2

def reid_feature_net(act='leaky'):
    return feature_net(act=act)
