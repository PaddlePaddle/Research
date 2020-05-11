from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


import os
import pdb

class feature_net_multi_branch():
    def __init__(self, act='leaky'):
        self.act = act

    def reid_classifier(self, input, fea_out_channel, class_num, name=None, is_train=True):
        cur_bran = fluid.layers.batch_norm(input=input, name=name + '_bn')
        cur_bran = fluid.layers.leaky_relu(x=cur_bran, alpha=0.1)
        cur_fea = fluid.layers.conv2d(input=cur_bran,num_filters=fea_out_channel,filter_size=1,bias_attr=False,name=name + '_bottleneck')
        if is_train:
            cur_fea_bn = fluid.layers.batch_norm(input=cur_fea, name=name + '_fea_bn')
            cur_fc = fluid.layers.fc(input=cur_fea_bn, size=class_num, name=name + '_fc')
            return cur_fea, cur_fc
        else:
            return cur_fea

    def net(self, input, is_train=False, num_features=256, class_dim=751, finetune=False):
        assert len(input) == 2
        x3, x4 = input
        
        x4_g_avg = fluid.layers.adaptive_pool2d(input=x4, pool_size=[1,1], pool_type='avg')
        x4_g_max = fluid.layers.adaptive_pool2d(input=x4, pool_size=[1,1], pool_type='max')
       
        x4_p_avg = fluid.layers.adaptive_pool2d(input=x4, pool_size=[2,2], pool_type='avg')
        x4_p_avg = fluid.layers.reshape(x4_p_avg, [x4_p_avg.shape[0], x4_p_avg.shape[1]*x4_p_avg.shape[2]*x4_p_avg.shape[3], 1, 1])
        x4_p_max = fluid.layers.adaptive_pool2d(input=x4, pool_size=[2,2], pool_type='max')
        x4_p_max = fluid.layers.reshape(x4_p_max, [x4_p_max.shape[0], x4_p_max.shape[1]*x4_p_max.shape[2]*x4_p_max.shape[3], 1, 1])

        x3_g_avg = fluid.layers.adaptive_pool2d(input=x3, pool_size=[1,1], pool_type='avg')
        x3_g_max = fluid.layers.adaptive_pool2d(input=x3, pool_size=[1,1], pool_type='max')
        
        x3_g_pool = x3_g_avg + x3_g_max
        x4_g_pool = x4_g_avg + x4_g_max
        x4_p_pool = x4_p_avg + x4_p_max

        if is_train:
            x3_g_pool_fea, x3_g_pool_fc = self.reid_classifier(x3_g_pool, num_features, class_dim, name='x3_g_pool') 
            x4_g_pool_fea, x4_g_pool_fc = self.reid_classifier(x4_g_pool, num_features, class_dim, name='x4_g_pool') 
            x4_p_pool_fea, x4_p_pool_fc = self.reid_classifier(x4_p_pool, num_features, class_dim, name='x4_p_pool')
            x3_g_avg = fluid.layers.squeeze(x3_g_avg, axes=[2,3])
            x3_g_max = fluid.layers.squeeze(x3_g_max, axes=[2,3])
            x4_g_avg = fluid.layers.squeeze(x4_g_avg, axes=[2,3])
            x4_g_max = fluid.layers.squeeze(x4_g_max, axes=[2,3])
            x4_p_avg = fluid.layers.squeeze(x4_p_avg, axes=[2,3])
            x4_p_max = fluid.layers.squeeze(x4_p_max, axes=[2,3])
            return x3_g_pool_fc, x4_g_pool_fc, x4_p_pool_fc, x3_g_avg, x3_g_max, x4_g_avg, x4_g_max, x4_p_avg, x4_p_max
        else:
            x3_g_pool_fea = self.reid_classifier(x3_g_pool, num_features, class_dim, name='x3_g_pool', is_train=is_train) 
            x4_g_pool_fea = self.reid_classifier(x4_g_pool, num_features, class_dim, name='x4_g_pool', is_train=is_train) 
            x4_p_pool_fea = self.reid_classifier(x4_p_pool, num_features, class_dim, name='x4_p_pool', is_train=is_train)
            x3_g_pool_fea = fluid.layers.squeeze(x3_g_pool_fea, axes=[2, 3])
            x4_g_pool_fea = fluid.layers.squeeze(x4_g_pool_fea, axes=[2, 3])
            x4_p_pool_fea = fluid.layers.squeeze(x4_p_pool_fea, axes=[2, 3])
            final_fea = fluid.layers.concat(input=[x3_g_pool_fea, x4_g_pool_fea, x4_p_pool_fea], axis=1)
            final_fea = fluid.layers.l2_normalize(x=final_fea, axis=1)
            return final_fea, final_fea 

def reid_feature_net_mb(act='leaky'):
    return feature_net_multi_branch(act='leaky')
