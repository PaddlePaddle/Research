import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout


class BASIC_BLOCK(fluid.dygraph.Layer):
    def __init__(self, num_channels=64, num_filters=64, pooltype=None):
        
        super(BASIC_BLOCK, self).__init__()
        self.conv = Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=3, stride=1, padding=1)
        self.batch_norm = BatchNorm(num_filters, act='relu')
        if pooltype:
            self.pooling = Pool2D(pool_size=2, pool_stride=2, pool_type=pooltype)
        else:
            self.pooling = None

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        if self.pooling:
            x = self.pooling(x)
        return x


class Conv4(fluid.dygraph.Layer):
    def __init__(self, args):

        super(Conv4, self).__init__()
        self.args = args
        if self.args.dataset == 'omniglot':
            input_channels = 1
        else:
            input_channels = 3
        if self.args.method == 'relationnet':
            pooltype = None
        else:
            pooltype = self.args.pooling_type
        self.conv0 = BASIC_BLOCK(num_channels=input_channels, num_filters=self.args.num_filters, pooltype=self.args.pooling_type)
        self.conv1 = BASIC_BLOCK(num_channels=self.args.num_filters, num_filters=self.args.num_filters, pooltype=self.args.pooling_type)
        self.conv2 = BASIC_BLOCK(num_channels=self.args.num_filters, num_filters=self.args.num_filters, pooltype=pooltype)
        self.conv3 = BASIC_BLOCK(num_channels=self.args.num_filters, num_filters=self.args.num_filters, pooltype=pooltype)
        if self.args.if_dropout:
            self.dropout = Dropout(p=0.5)

    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.conv1(x)
        if self.args.if_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.args.if_dropout:
            x = self.dropout(x)
        x = self.conv3(x)
        return x