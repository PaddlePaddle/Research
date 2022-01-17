#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: base_model.py
func: custom resnet50 defined in tf2.0
Describe: add CBAM and multi scale feature based on resnet50
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/07/21
"""
import paddle
import paddle.nn as nn
import numpy as np

from mmflib.arch.gears.mgd import MultiGlobalDes
from mmflib.arch.gears.neck import BottleNeck

print("Paddle Version: ", paddle.__version__)

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']

# BACKBONE_PATH = "/home/yuwei/code/paddle/arcface/best_model/custom_res50.pth"
# MULTI_DES_PATH = "/home/yuwei/code/paddle/arcface/best_model/fuselayer.pth"
# BOTTLENECK_PATH =  "/home/yuwei/code/paddle/arcface/best_model/bottlenecklayer.pth"

class ChannelAttention(nn.Layer):
    """channel attention
    Args:
        in_planes: 输入特征的通道数
        ratio：输出通道数对应的比例
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
           
        self.fc = nn.Sequential(nn.Conv2D(in_planes, in_planes // 16, 1, bias_attr=False),
                               nn.ReLU(),
                               nn.Conv2D(in_planes // 16, in_planes, 1, bias_attr=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """forward"""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Layer):
    """spatial attention
    Args:
        kernel_size: 卷积核大小
    """
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=kernel_size // 2, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """forward"""
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Layer):
    """CBAM
    Args:
        in_planes: 输入特征通道数
    """
    def __init__(self, in_planes):
        super(CBAM, self).__init__()

        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_planes)

    def forward(self, x):
        """forward"""
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x


def Conv1(in_planes, places, stride=2):
    """conv1
    Args:
        in_planes: 输入特征通道数
        places: 输出特征通道数
        stride： 步长
    Returns:
        卷积操作
    """
    return nn.Sequential(
        nn.Conv2D(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias_attr=True),
        nn.BatchNorm2D(places),
        nn.ReLU(),
        nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
    )


def down_layer(in_planes, places, stride=2, kernel_size=3, padding=1):
    """down layer
    Args:
        in_planes: 输入特征通道数
        places: 输出特征通道数
        stride： 步长
        kernel_size： 卷积大小
        padding： padding大小
    Returns:
        降采样操作
    """
    return nn.Sequential(
        nn.Conv2D(in_channels=in_planes, 
                  out_channels=places, 
                  kernel_size=kernel_size, 
                  stride=stride, 
                  padding=padding, 
                  bias_attr=True),
        nn.BatchNorm2D(places),
        nn.ReLU()
    )


class Bottleneck(nn.Layer):
    """bottleneck
    Args:
        in_planes: 输入特征通道数
        places: 输出特征通道数
        stride： 步长
        downsampling： 是否降采样
        expansion： 通道数比例
    """
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2D(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias_attr=True),
            nn.BatchNorm2D(places),
            nn.ReLU(),
            nn.Conv2D(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias_attr=True),
            nn.BatchNorm2D(places),
            nn.ReLU(),
            nn.Conv2D(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias_attr=True),
            nn.BatchNorm2D(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels=in_places, 
                          out_channels=places * self.expansion, 
                          kernel_size=1, 
                          stride=stride, 
                          bias_attr=True),
                nn.BatchNorm2D(places * self.expansion)
            )
        self.relu = nn.ReLU()
        self.cbam = CBAM(places * self.expansion)
 
    def forward(self, x):
        """forward"""
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out = self.cbam(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Layer):
    """resnet
    Args:
        blocks: 每个残差块对应的blok个数
        num_classes： 分类对应的类别数
        expansion： 通道数比例
    """
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.padding1 = nn.Pad2D(3)
        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=1)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)
        self.down1 = down_layer(in_planes=256, places=512, stride=4)
        self.down2 = down_layer(in_planes=512, places=512, stride=2)
        self.down3 = down_layer(in_planes=1024, places=512, stride=2)
        self.down4 = down_layer(in_planes=2048, places=512, stride=1, kernel_size=1, padding=0)
        

    def make_layer(self, in_places, places, block, stride):
        """make layer
        Args:
            in_planes: 输入特征通道数
            places: 输出特征通道数
            block: block个数
            stride： 步长
        """
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        """forward"""
        x = self.conv1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x1 = self.down1(x1)
        x2 = self.down2(x2)
        x3 = self.down3(x3)
        x4 = self.down4(x4)

        x = paddle.concat([x1, x2, x3, x4], axis=1)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class ResNetCustom(nn.Layer):
    """resnet50 custom
    Args:
        num_bottleneck: 输出特征维度
        droprate： dropout比例
        stop_layer： 输出层 ['feamap', 'neck']
    """
    def __init__(self, num_bottleneck, droprate=0.5, stop_layer=None):
        super(ResNetCustom, self).__init__()
        assert stop_layer in ['feamap', 'neck']
        self.backbone = ResNet50()
        #self.backbone.load_state_dict(torch.load(BACKBONE_PATH)['model_state'])
        fea_len = 2048
        self.multi_des = MultiGlobalDes(fea_len, drop_rate=droprate)
        #self.multi_des.load_state_dict(torch.load(MULTI_DES_PATH)['model_state'])

        fea_len = num_bottleneck * 3
        self.bottleneck = BottleNeck(fea_len, num_bottleneck, droprate=droprate, inp_bn=True)
        #self.bottleneck.load_state_dict(torch.load(BOTTLENECK_PATH)['model_state'])
        self.stop_layer = stop_layer

    def forward(self, x):
        """forward"""
        x = self.backbone(x)
        if self.stop_layer == 'feamap':
            return x
        x = self.multi_des(x)
        x = self.bottleneck(x)

        return x


def ResNet50():
    """resnet50"""
    return ResNet([3, 4, 6, 3])


def ResNet101():
    """resnet101"""
    return ResNet([3, 4, 23, 3])
    

def ResNet152():
    """resnet152"""
    return ResNet([3, 8, 36, 3])


if __name__ == '__main__':
    model = ResNetCustom(256, stop_layer='neck')
    print(paddle.summary(model, (2, 3, 224, 224)))
    # input = paddle.randn(shape=[2, 3, 224, 224])
    # out = model(input)
    # print(out.shape)

    #checkpoint = torch.load("/data/vit_feature_best/arcface/model_33_840000.pth", map_location=torch.device('cpu'))
    ###仅加载backbone参数
    #model_dict = model.state_dict()
    #state_dict = {k:v for k, v in checkpoint['model_state'].items() if 'head' not in k and 'bottleneck' not in k}
    #state_dict = {k:v for k, v in checkpoint['model_state'].items() if 'head' not in k and 'bottleneck' not in k and 'multi_des' not in k} 
    # state_dict = {k:v for k, v in checkpoint['model_state'].items() if 'head' not in k}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)