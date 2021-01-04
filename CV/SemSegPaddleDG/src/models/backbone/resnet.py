from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import paddle.fluid as fluid
import os
import paddle.nn as nn

__all__ = [
    "ResNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"
]

class ConvBN(fluid.dygraph.Layer):

    def __init__(self,
                 name_scope,
                 input_channels,
                 num_filters,
                 filter_size=3,
                 stride=1,
                 dilation=1,
                 act=None,
                 learning_rate=1.0,
                 dtype='float32',
                 bias_attr=False):
        super(ConvBN, self).__init__(name_scope)

        if dilation != 1:
            padding = dilation
        else:
            padding = (filter_size - 1) // 2
        self._conv = nn.Conv2D(in_channels=input_channels, 
                                          out_channels=num_filters,
                                          kernel_size=filter_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          bias_attr=bias_attr if bias_attr is False else fluid.ParamAttr(
                                              learning_rate=learning_rate,
                                              trainable=True),
                                          weight_attr=fluid.ParamAttr(
                                              learning_rate=learning_rate,
                                              trainable=True)
                                          )
        self._bn = nn.SyncBatchNorm(num_filters,
                                    momentum=0.9,
                                    epsilon=1e-5,
                                    bias_attr=fluid.ParamAttr(
                                        learning_rate=learning_rate,
                                        trainable=True),
                                    weight_attr=fluid.ParamAttr(
                                        learning_rate=learning_rate,
                                        trainable=True))
        self.act = act

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        if self.act is not None:
            x = nn.ReLU()(x)
        return x


class BottleneckBlock(fluid.dygraph.Layer):
    expansion = 4
    def __init__(self, name_scope, input_channels, num_filters, stride, dilation=1, same=False):
        super(BottleneckBlock, self).__init__(name_scope)

        self._conv0 = ConvBN(name_scope,
                             input_channels=input_channels,
                             num_filters=num_filters,
                             filter_size=1,
                             stride=1,
                             act='relu')
        self._conv1 = ConvBN(name_scope,
                             input_channels=num_filters,
                             num_filters=num_filters,
                             filter_size=3,
                             stride=stride,
                             dilation=dilation,
                             act='relu')
        self._conv2 = ConvBN(name_scope,
                             input_channels=num_filters,
                             num_filters=num_filters * self.expansion,
                             filter_size=1,
                             stride=1,
                             act=None)
        self.same = same

        if not same:
            self._skip = ConvBN(name_scope,
                                input_channels=input_channels,
                                num_filters=num_filters * self.expansion,
                                filter_size=1,
                                stride=stride,
                                act=None)

    def forward(self, inputs):
        x = self._conv0(inputs)
        x = self._conv1(x)
        x = self._conv2(x)
        if self.same:
            skip = inputs
        else:
            skip = self._skip(inputs)
        x = fluid.layers.elementwise_add(x, skip, act='relu')
        return x


class ResNet(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 layer=152,
                 num_class=1000,
                 dilated=True,
                 multi_grid=False,
                 multi_dilation=[2, 4, 8],
                 need_fc=False):
        super(ResNet, self).__init__(name_scope)

        self.inplanes = 64
        support_layer = [18, 34, 50, 101, 152]
        assert layer in support_layer, 'layer({}) not in {}'.format(layer, support_layer)
        self.need_fc = need_fc
        self.num_filters_list = [64, 128, 256, 512]
        if layer == 18:
            self.depth = [2, 2, 2, 2]
        elif layer == 34:
            self.depth = [3, 4, 6, 3]
        elif layer == 50:
            self.depth = [3, 4, 6, 3]
        elif layer == 101:
            self.depth = [3, 4, 23, 3]
        elif layer == 152:
            self.depth = [3, 8, 36, 3]
        print('multi_grid:', multi_grid)

        if multi_grid:
            assert multi_dilation is not None
            print('-------------use_multi_grid------------')
            self.multi_dilation = multi_dilation

        self._conv = ConvBN(name_scope, 3, 64, 7, 2, act='relu')
        self._pool = fluid.dygraph.Pool2D(pool_size=3,
                                          pool_stride=2,
                                          pool_padding=1,
                                          pool_type='max')
        if layer >= 50:
            self.layer1 = self._make_layer(block=BottleneckBlock,
                                           depth=self.depth[0],
                                           num_filters=self.num_filters_list[0],
                                           stride=1,
                                           same=False,
                                           name='layer1')
            self.layer2 = self._make_layer(block=BottleneckBlock,
                                           depth=self.depth[1],
                                           num_filters=self.num_filters_list[1],
                                           stride=2,
                                           same=False,
                                           name='layer2')
            if dilated:
                self.layer3 = self._make_layer(block=BottleneckBlock,
                                               depth=self.depth[2],
                                               num_filters=self.num_filters_list[2],
                                               stride=1,
                                               dilation=2,
                                               same=False,
                                               name='layer3')
                if multi_grid:  # layer4 采用不同的采样率
                    self.layer4 = self._make_layer(block=BottleneckBlock,
                                                   depth=self.depth[3],
                                                   num_filters=self.num_filters_list[3],
                                                   stride=2,
                                                   dilation=4,
                                                   multi_grid=multi_grid,
                                                   multi_dilation=self.multi_dilation,
                                                   same=False,
                                                   name='layer4')
                else:
                    self.layer4 = self._make_layer(block=BottleneckBlock,
                                                   depth=self.depth[3],
                                                   num_filters=self.num_filters_list[3],
                                                   stride=1,
                                                   dilation=4,
                                                   same=False,
                                                   name='layer4')
            else:
                self.layer3 = self._make_layer(block=BottleneckBlock,
                                               depth=self.depth[2],
                                               num_filters=self.num_filters_list[2],
                                               stride=2,
                                               dilation=1,
                                               same=False,
                                               name='layer3')
                self.layer4 = self._make_layer(block=BottleneckBlock,
                                               depth=self.depth[3],
                                               num_filters=self.num_filters_list[3],
                                               stride=2,
                                               dilation=1,
                                               same=False,
                                               name='layer4')

        else:  # layer=18 or layer=34
            self.layer1 = self._make_layer(block=BasicBlock,
                                           depth=self.depth[0],
                                           num_filters=self.num_filters_list[0],
                                           stride=1,
                                           same=True,
                                           name=name_scope)
            self.layer2 = self._make_layer(block=BasicBlock,
                                           depth=self.depth[1],
                                           num_filters=self.num_filters_list[1],
                                           stride=2,
                                           same=False,
                                           name=name_scope)
            self.layer3 = self._make_layer(block=BasicBlock,
                                           depth=self.depth[2],
                                           num_filters=self.num_filters_list[2],
                                           stride=2,
                                           dilation=1,
                                           same=False,
                                           name=name_scope)
            self.layer4 = self._make_layer(block=BasicBlock,
                                           depth=self.depth[3],
                                           num_filters=self.num_filters_list[3],
                                           stride=2,
                                           dilation=1,
                                           same=False,
                                           name=name_scope)

        self._avgpool = fluid.dygraph.Pool2D(pool_size=-1,
                                             global_pooling=True,
                                             pool_type='avg')
        self.fc = fluid.dygraph.Linear(input_dim=self.num_filters_list[-1] * BottleneckBlock.expansion,
                                   output_dim=num_class,
                                   act=None)

    def _make_layer(self, block, depth, num_filters, stride=1, dilation=1, same=False, multi_grid=False,
                    multi_dilation=None, name=None):
        layers = []
        if dilation != 1:
            #  stride(2x2) with a dilated convolution instead
            stride = 1
        
        if multi_grid:
            assert len(multi_dilation) == 3
            temp = block(name + '.{}'.format(0),
                             input_channels=self.inplanes,
                             num_filters=num_filters,
                             stride=stride,
                             dilation=multi_dilation[0],
                             same=same)
            same = True
            self.inplanes = num_filters * block.expansion
            layers.append(self.add_sublayer('_{}_{}'.format(name, 0), temp))
            for depth in range(1, depth):
                temp = block(name + '.{}'.format(depth),
                             input_channels=self.inplanes,
                             num_filters=num_filters,
                             stride=stride,
                             dilation=multi_dilation[depth],
                             same=same)
                layers.append(self.add_sublayer('_{}_{}'.format(name, depth + 1), temp))
        else:
            temp = block(name + '.{}'.format(0),
                             input_channels=self.inplanes,
                             num_filters=num_filters,
                             stride=stride,
                             dilation= int(dilation/2) if dilation > 1 else 1,
                             same=same)
            same = True
            self.inplanes = num_filters * block.expansion
            layers.append(self.add_sublayer('_{}_{}'.format(name, 0), temp))
            for depth in range(1, depth):
                temp = block(name + '.{}'.format(depth),
                             input_channels=self.inplanes,
                             num_filters=num_filters,
                             stride=1,
                             dilation=dilation,
                             same=same)
                
                layers.append(self.add_sublayer('_{}_{}'.format(name, depth + 1), temp))
        return layers

    def forward(self, inputs):
        x = self._conv(inputs)

        x = self._pool(x)
        for layer in self.layer1:
            x = layer(x)
        c1 = x

        for layer in self.layer2:
            x = layer(x)
        c2 = x

        for layer in self.layer3:
            x = layer(x)
        c3 = x

        for layer in self.layer4:
            x = layer(x)
        c4 = x

        if self.need_fc:
            x = self._avgpool(x)
            x = self.fc(x)
            return x
        else:
            return c1, c2, c3, c4

def ResNet18():
    model = ResNet('resnet18',layer=18)
    return model


def ResNet34():
    model = ResNet('resnet34',layer=34)
    return model


def ResNet50():
    model = ResNet('resnet50',layer=50)
    return model


def ResNet101(pretrained='pretrained_model/resnet101_pretrained.pdparams', multi_grid=False):
#def ResNet101(pretrained=None, multi_grid=False):
    model = ResNet('resnet101', layer=101, multi_grid=multi_grid)
    if pretrained is not None:
        param, _ = fluid.load_dygraph(pretrained)
        model.set_dict(param)
        print('successfully load the pretrained_model')
    return model


def ResNet152():
    model = ResNet('resnet152',layer=152)
    return model

if __name__ == '__main__':
    import numpy as np
    with fluid.dygraph.guard():
        model=ResNet101()
        x = np.random.randn(2, 3, 768, 768).astype('float32')
        x = fluid.dygraph.to_variable(x)
        c1,c2, c3, c4 = model(x)
        # num_param = 0
        # for param in model.parameters():
        #     temp = 1
        #     for j in range(len(param.shape)):
        #         temp = temp * param.shape[j]
        #     num_param += temp*4
        # print('model_params:', num_param/1e6, ' MB')
        print(c1.shape)
        
