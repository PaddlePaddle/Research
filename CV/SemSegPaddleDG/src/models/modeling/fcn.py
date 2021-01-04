from __future__ import division
from __future__ import print_function
import sys
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from src.models.backbone.resnet import ResNet

from src.utils.config import cfg

    

class FCN_Head(fluid.dygraph.Layer):
    def __init__(self, input_channels, out_channels):
        super(FCN_Head, self).__init__()
        inter_channels = input_channels // 4
        self.conv5 = fluid.dygraph.Conv2D(input_channels, inter_channels, 3, stride=1, padding=1, bias_attr=False)
        self.bn = fluid.dygraph.BatchNorm(inter_channels, act='relu')
        self.Dropout = fluid.dygraph.Dropout(p=0.1)
        self.conv6 = fluid.dygraph.Conv2D(inter_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv5(x)
        x = self.bn(x)
        x = self.Dropout(x)
        x = self.conv6(x)
        return x



class FCN(fluid.dygraph.Layer):
    def __init__(self, input_channels=3, num_classes=19):
        super(FCN, self).__init__()
        self.resnet = ResNet('resnet101', layer=101)
        self.num_classes = num_classes
        self.head = FCN_Head(2048, self.num_classes)
        self.auxlayer = FCN_Head(1024, self.num_classes)

    def forward(self, input):
        N, C, H, W = input.shape
        _, _, res4, res5 = self.resnet(input)
        x = self.head(res5)
        x = fluid.layers.interpolate(x, out_shape=[H, W])
        
        auxout = self.auxlayer(res4)
        auxout = fluid.layers.interpolate(auxout, out_shape=[H, W])
        return x, auxout


if __name__ == "__main__":
    from paddle.fluid.dygraph.base import to_variable
    import numpy as np
    with fluid.dygraph.guard():
        model = GruModule(input_channel=512, num_state=128, num_node=64)
        data = np.random.uniform(-1, 1, [2, 512, 96, 96]).astype('float32')
        data = to_variable(data)
        y = model(data)
        print(y.shape)

        


