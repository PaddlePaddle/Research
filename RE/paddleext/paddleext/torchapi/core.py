"""
paddle core
"""
import sys
import types
from functools import partial
from types import MethodType
from typing import Any

import paddle
import random
import numpy as np

Module = paddle.nn.Layer
ModuleBase = paddle.nn.Layer
ModuleDict = paddle.nn.LayerDict
ModuleList = paddle.nn.LayerList
device=str

dtype=paddle.dtype

def load_state_dict(module: Module, state_dict, *args, **kwargs):
    module.set_state_dict(state_dict, *args, **kwargs)


Module.load_state_dict = load_state_dict

from paddle import *

def deterministic(seed=0):
    seed = 0
    random.seed(seed)
    paddle.seed(seed)
    np.random.seed(seed)


import paddle

from paddle import bool, int32, int64, int8, float32, float64, float16

long = paddle.int64
int = paddle.int32
float = paddle.float32
double = paddle.float64


def platform():
    """

    Returns:

    """

    return "paddle"



from paddle import no_grad, autograd

class set_detect_anomaly(object):
    r"""Context-manager that sets the anomaly detection for the autograd engine on or off.
    ``set_detect_anomaly`` will enable or disable the autograd anomaly detection
    based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.
    See ``detect_anomaly`` above for details of the anomaly detection behaviour.
    Args:
        mode (bool): Flag whether to enable anomaly detection (``True``),
                     or disable (``False``).
    """

    def __init__(self, mode: bool) -> None:
        pass

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> None:
        pass


setattr(autograd, "set_detect_anomaly", set_detect_anomaly)


def paddle_delegate_func(func, *args, **kwargs):
    if "dim" in kwargs:
        kwargs["axis"] = kwargs["dim"]
        del kwargs["dim"]

    if "device" in kwargs:
        del kwargs["device"]

    return func(*args, **kwargs)

def make_delegate_class(class_):

    class DelegateClass(class_):
        def __init__(self, *args, **kwargs):

            if class_.__name__.endswith("Linear"):
                if "bias" in kwargs:
                    kwargs["bias_attr"] = kwargs["bias"]
                    del kwargs["bias"]
                if "weight" in kwargs:
                    kwargs["weight_attr"] = kwargs["weight"]
                    del kwargs["weight"]
            if class_.__name__.endswith("LayerNorm"):
                if "eps" in kwargs:
                    kwargs["epsilon"] = kwargs["eps"]
                    del kwargs["eps"]
            super().__init__(*args, **kwargs)
#            self.__class__ = class_

    return DelegateClass


