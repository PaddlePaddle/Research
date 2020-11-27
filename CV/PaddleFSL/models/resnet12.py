import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout

class BASIC_BLOCK(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, padding):
        
        super(BASIC_BLOCK, self).__init__()
        self.conv = Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=filter_size, stride=1, padding=padding)
        self.batch_norm = BatchNorm(num_filters, act=None)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        return x

class Residual_Block(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, pooltype):

        super(Residual_Block, self).__init__()
        self.short_cut = BASIC_BLOCK(num_channels=num_channels, num_filters=num_filters, filter_size=1, padding=0)
        self.conv0 = BASIC_BLOCK(num_channels=num_channels, num_filters=num_filters, filter_size=3, padding=1)
        self.conv1 = BASIC_BLOCK(num_channels=num_filters, num_filters=num_filters, filter_size=3, padding=1)
        self.conv2 = BASIC_BLOCK(num_channels=num_filters, num_filters=num_filters, filter_size=3, padding=1)
        if pooltype:
            self.pooling = Pool2D(pool_size=2, pool_stride=2, pool_type=pooltype)
        else:
            self.pooling = None

    def forward(self, inputs):
        short_cut = self.short_cut(inputs)
        x = self.conv0(inputs)
        x = fluid.layers.leaky_relu(x, alpha=0.1)
        x = self.conv1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.1)
        x = self.conv2(x)
        x += short_cut
        x = fluid.layers.leaky_relu(x, alpha=0.1)
        if self.pooling:
            x = self.pooling(x)
        return x

class ResNet12(fluid.dygraph.Layer):
    def __init__(self, args):

        super(ResNet12, self).__init__()
        self.args = args
        if self.args.dataset == 'omniglot':
            input_channels = 1
        else:
            input_channels = 3
        if self.args.method == 'relationnet':
            pooltype = None
        else:
            pooltype = self.args.pooling_type
        self.res_block0 = Residual_Block(num_channels=input_channels, num_filters=self.args.resnet12_num_filters[0], pooltype=self.args.pooling_type)
        self.res_block1 = Residual_Block(num_channels=self.args.resnet12_num_filters[0], num_filters=self.args.resnet12_num_filters[1], pooltype=self.args.pooling_type)
        self.res_block2 = Residual_Block(num_channels=self.args.resnet12_num_filters[1], num_filters=self.args.resnet12_num_filters[2], pooltype=pooltype)
        self.res_block3 = Residual_Block(num_channels=self.args.resnet12_num_filters[2], num_filters=self.args.resnet12_num_filters[3], pooltype=pooltype)
        self.gap = Pool2D(pool_type='avg', global_pooling=True)
        if self.args.if_dropout: 
            self.dropout = Dropout(p=0.5)

    def forward(self, inputs):
        x = self.res_block0(inputs)
        x = self.res_block1(x)
        if self.args.if_dropout:
            x = self.dropout(x)
        x = self.res_block2(x) 
        if self.args.if_dropout:
            x = self.dropout(x)
        x = self.res_block3(x)
        if self.args.method != 'relationnet':
            x = self.gap(x)

        return x