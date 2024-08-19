import paddle
import random
import numpy as np

ModuleBase = paddle.nn.Layer
ModuleDict = paddle.nn.LayerDict
ModuleList = paddle.nn.LayerList

from paddle.nn import *

Conv2d = Conv2D
Conv3d = Conv3D
from . import functional
from paddle.nn import initializer

from . import init

def Parameter(data, requires_grad=True):
    """

    Args:
        data:
        requires_grad:

    Returns:

    """

    param = paddle.create_parameter(data.shape, dtype=data.dtype, default_initializer=initializer.Assign(data))

    param.stop_gradient = not requires_grad

    return param

from paddle.fluid import framework

class Module(paddle.nn.Layer):
    """
    Module with add_parameter
    """

    def __setattr__(self, key, value):

        if isinstance(value, framework.Parameter):
            self.add_parameter(key, value)
        else:
            super().__setattr__(key, value)