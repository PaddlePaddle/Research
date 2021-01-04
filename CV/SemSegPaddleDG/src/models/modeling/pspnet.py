from __future__ import division
from __future__ import print_function
import sys
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
import paddle.nn.functional as F
import paddle.nn as nn
from src.models.backbone.resnet import ResNet101
from src.utils.config import cfg


class MakePath(fluid.dygraph.Layer):
    def __init__(self, name_scope, fea_dim, reduction_dim):
        super(MakePath, self).__init__(name_scope)
        #conv2d\bn\Relu
        learning_rate = 10.0
        assert reduction_dim == int(reduction_dim), 'the input dims of psphead cannot be fully devided by path number, please check!'
        reduction_dim = int(reduction_dim)
        self.conv = fluid.dygraph.Conv2D(fea_dim,
                                         reduction_dim,
                                         1, 
                                         1, 
                                         bias_attr=False,
                                         param_attr=fluid.ParamAttr(
                                              learning_rate=learning_rate,
                                              trainable=True))
        self.bn = nn.SyncBatchNorm(reduction_dim, 
                                   bias_attr   = fluid.ParamAttr(learning_rate=learning_rate, trainable=True),
                                   weight_attr = fluid.ParamAttr(learning_rate=learning_rate, trainable=True)
                  )
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PSPHead(fluid.dygraph.Layer):
    # Arch of Pyramid Scene Parsing Module:                                                 
    #
    #          |----> Pool_1x1 + Conv_1x1 + BN + ReLU + bilinear_interp-------->|————————|
    #          |                                                                |        |
    #          |----> Pool_2x2 + Conv_1x1 + BN + ReLU + bilinear_interp-------->|        | 
    # x ------>|                                                                | concat |----> Conv_3x3 + BN + ReLU -->Dropout --> Conv_1x1
    #     |    |----> Pool_3x3 + Conv_1x1 + BN + ReLU + bilinear_interp-------->|        | 
    #     |    |                                                                |        |
    #     |    |----> Pool_6x6 + Conv_1x1 + BN + ReLU + bilinear_interp-------->|________|
    #     |                                                                              ^
    #     |——————————————————————————————————————————————————————————————————————————————|
    #

    def __init__(self,
                 fea_dim,
                 num_classes,
                 name_scope=''):
    
        super(PSPHead, self).__init__(name_scope)
        self.num_classes=num_classes
        self.sizes=(1,2,3,6)
        self.layers=[]
        learning_rate = 10.0

        for size in self.sizes:
            path = MakePath('psp_' + str(size), fea_dim, fea_dim/len(self.sizes))
            self.layers.append(self.add_sublayer('_{}'.format('psp_' + str(size) + '_path'), path))

        fea_dim = fea_dim * 2
        self.pre_cls_conv = fluid.dygraph.Conv2D(fea_dim, 
                                                 512, 
                                                 filter_size=3, 
                                                 stride=1, 
                                                 padding=1, 
                                                 bias_attr=False, 
                                                 param_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True))
        self.pre_cls_bn = nn.SyncBatchNorm(512, 
                                            weight_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True),
                                            bias_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True))
        self.pre_drop_out = fluid.dygraph.Dropout(p=0.1)
        self.cls_conv = fluid.dygraph.Conv2D(512, num_classes, filter_size=1, param_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True),
                                                  bias_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True),)


    def forward(self, inputs):
        cat_layers=[inputs]
        n, c, h, w = inputs.shape
        for size in range(len(self.sizes)):
            pool_kernel_h = int(h / self.sizes[size]) + 1
            pool_kernel_w = int(w / self.sizes[size]) + 1
            padding_h = pool_kernel_h * self.sizes[size] - h
            padding_w = pool_kernel_w * self.sizes[size] - w
            pool_feat = fluid.dygraph.Pool2D(pool_size=[pool_kernel_h, pool_kernel_w], 
                                             pool_type='avg', 
                                             pool_stride=[pool_kernel_h, pool_kernel_w],
                                             pool_padding=[padding_h, padding_w])(inputs)
            sub_path = self.layers[size](pool_feat)
            interpolate = F.common.interpolate(sub_path, size=[h, w], mode='BILINEAR')
            cat_layers.append(interpolate)
        cat_feat = fluid.layers.concat(input=cat_layers, axis=1)
        feature = self.pre_cls_conv(cat_feat)
        feature = self.pre_cls_bn(feature)
        feature = nn.ReLU()(feature)
        feature = self.pre_drop_out(feature)
        feature = self.cls_conv(feature)

        return feature


class AuxHead(fluid.dygraph.Layer):
    def __init__(self, input_dims, num_classes, drop_out=0.1, name_scope=''):
        super(AuxHead, self).__init__(name_scope)
        learning_rate = 10.0
        self.conv_1 = fluid.dygraph.Conv2D(input_dims, 256, filter_size=3, padding=1, bias_attr=False, param_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True))
        self.bn = nn.SyncBatchNorm(256, weight_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True),
                                                  bias_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True))
        self.dropout = fluid.dygraph.Dropout(p=drop_out)
        self.conv_2 = fluid.dygraph.Conv2D(256, num_classes, filter_size=1, param_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True),
                                                  bias_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True))
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        return x



class PSPNet(fluid.dygraph.Layer):
    """
    Reference: 
        Zhao, Hengshuang, et al. "Pyramid scene parsing network.", In CVPR 2017
    """
    def __init__(self, pretrained, input_channels=3, num_classes=50, name_scope='', multi_grid=True):
        super(PSPNet, self).__init__(name_scope)
        if 'resnet' in cfg.MODEL.BACKBONE:
            assert input_channels==3, 'input_channel is not 3, please check!'
            self.resnet = ResNet101(pretrained= pretrained, multi_grid=multi_grid)
        elif 'hrnet' in cfg.MODEL.BACKBONE:
            self.hrnet=hrnet(input)
        else:
            raise Exception("pspnet only support resnet and hrnet backbone")
        self.num_classes=num_classes
        self.psphead=PSPHead(2048, self.num_classes)
        self.auxhead=AuxHead(1024, self.num_classes)
    def forward(self, inputs):
        if 'resnet' in cfg.MODEL.BACKBONE:
            c1, c2, res4, res5 = self.resnet(inputs)
        elif 'hrnet' in cfg.MODEL.BACKBONE:
            res5 = self.hrnet
        else:
            raise Exception("pspnet only support resnet and hrnet backbone")

        logit = self.psphead(res5)
        seg_name = "logit"
        shape = inputs.shape
        logit = F.common.interpolate(logit, size=shape[2:], mode='BILINEAR')

        if cfg.MODEL.PSPNET.AuxHead:
            aux_logit = self.auxhead(res4)
            aux_seg_name= "Aux_layer2"
            aux_logit=F.common.interpolate(aux_logit, size=shape[2:], mode='BILINEAR')
            return logit, aux_logit
        return logit

if __name__ == '__main__':
    from paddle.fluid.dygraph.base import to_variable
    import numpy as np
    with fluid.dygraph.guard():
        model=PSPHead(fea_dim=2048, num_classes=19)
        data = np.random.uniform(-1, 1, [1, 2048, 28, 28]).astype('float32')
        data = to_variable(data)
        y=model(data)
        print(y.shape)
        
        
