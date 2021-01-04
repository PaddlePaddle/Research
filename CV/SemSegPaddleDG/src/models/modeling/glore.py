from __future__ import division
from __future__ import print_function
import sys
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.param_attr import ParamAttr

from src.models.backbone.resnet import ResNet101

from src.utils.config import cfg
from src.models.modeling.pspnet import AuxHead


class GCN(nn.Layer):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1D(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1D(num_state, num_state, kernel_size=1, bias_attr=bias)

    def forward(self, x):
        h = self.conv1(fluid.layers.transpose(x, perm=(0, 2, 1)))
        h = fluid.layers.transpose(h, perm=(0, 2, 1))
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class GruModule(fluid.dygraph.Layer):
    def __init__(self, input_channel=512, num_state=128, num_node=64, normalize=False):
        super(GruModule, self).__init__()
        self.Normalize = normalize
        self.num_state = 128 #  node channels
        self.num_node = 64 # node number
        self.reduction_dim = fluid.dygraph.Conv2D(input_channel, num_state, filter_size=1)
        self.projection_mat = fluid.dygraph.Conv2D(input_channel, num_node, filter_size=1)
        self.gcn = GCN(num_state=self.num_state, num_node=self.num_node)
        self.extend_dim = fluid.dygraph.Conv2D(self.num_state, input_channel,filter_size=1, bias_attr=False)
        self.extend_bn = nn.SyncBatchNorm(input_channel, epsilon=1e-4)


    def forward(self, input):
        n, c, h, w = input.shape
        reduction_dim = self.reduction_dim(input) # B, C, H, W
        mat_B = self.projection_mat(input) # B, N, H, W
        
        reshaped_reduction = fluid.layers.reshape(reduction_dim, shape=[n, self.num_state, h * w])# B, C, H*W
        # transposed_reduce = fluid.layers.transpose(reshaped_reduction, perm=[0, 2, 1]) # B, H*W, C
        
        reshaped_B = fluid.layers.reshape(mat_B, shape=[n, self.num_node, h * w]) # B, N, H*W

        reproject = reshaped_B # B, N, H*W
        
        node_state_V = fluid.layers.matmul(reshaped_reduction, fluid.layers.transpose(reshaped_B, perm=[0, 2, 1])) # B, C, N

        if self.Normalize:
            dom = fluid.layers.fill_constant(shape=[1], value=h*w, dtype='float32')
            node_state_V = fluid.layers.elementwise_div(node_state_V, dom)
        gcn_out = self.gcn(node_state_V)# B, C, N
        Y = fluid.layers.matmul(gcn_out, reproject)# B, C, H*W
        Y = fluid.layers.reshape(Y, shape=[n, self.num_state, h, w])# B, C, H, W
        Y_extend = self.extend_dim(Y)
        Y_extend = self.extend_bn(Y_extend)

        out = fluid.layers.elementwise_add(input, Y_extend)
        return out
    

class Glore(fluid.dygraph.Layer):
    """
    Reference:
       Chen, Yunpeng, et al. "Graph-Based Global Reasoning Networks", In CVPR 2019
    """
    def __init__(self, pretrained, input_channels=3, num_classes=19, multi_grid=False):
        super(Glore, self).__init__()
        self.resnet = ResNet101(pretrained = pretrained, multi_grid=True)
        self.num_classes = num_classes
        self.gru_module = GruModule(input_channel=512, num_state=128, num_node=64)
        self.auxhead = AuxHead(1024, self.num_classes)
        self.dropout = fluid.dygraph.Dropout(0.1)
        learning_rate = 10.0
        if cfg.DATASET.DATASET_NAME=='cityscapes':
            self.connect_conv = fluid.dygraph.Conv2D(2048, 512, 1, bias_attr=False, param_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True))
        else:
            self.connect_conv = fluid.dygraph.Conv2D(2048, 512, 3, bias_attr=False, param_attr=fluid.ParamAttr(learning_rate=learning_rate, trainable=True))
        self.connect_bn = nn.SyncBatchNorm(512)
        self.connect_relu = nn.ReLU()
        param_attr = fluid.ParamAttr(regularizer= fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0),
                                     initializer= fluid.initializer.TruncatedNormal(loc=0.0, scale=0.01),
                                     learning_rate=10.0)
        self.get_logit_conv = fluid.dygraph.Conv2D(512, num_classes, filter_size=1, param_attr=param_attr, bias_attr=True)

    def forward(self, input):
        N, C, H, W = input.shape
        _, _, res4, res5 = self.resnet(input)
        
        feature = self.connect_conv(res5)
        feature = self.connect_bn(feature)
        feature = self.connect_relu(feature)
        gru_output = self.gru_module(feature)
        dropout = self.dropout(gru_output)
        logit = self.get_logit_conv(dropout)
        logit = F.common.interpolate(logit, size=[H, W], mode='BILINEAR')
        
        if 1:
            aux_logit = self.auxhead(res4)
            aux_logit = F.common.interpolate(aux_logit, size=[H, W], mode='BILINEAR')
            return logit, aux_logit
        return logit


if __name__ == "__main__":
    from paddle.fluid.dygraph.base import to_variable
    import numpy as np
    with fluid.dygraph.guard():
        model = GruModule(input_channel=512, num_state=128, num_node=64)
        data = np.random.uniform(-1, 1, [2, 512, 96, 96]).astype('float32')
        data = to_variable(data)
        y = model(data)
        print(y.shape)

        


