from functools import partial

from paddle.optimizer.lr import *

StepLR = StepDecay
ExponentialLR = ExponentialDecay

#
# def paddle_lr_class_creator(paddle_lr_class, *args, **kwargs):
#
#     if "lr" in kwargs:
#         kwargs["learning_rate"] = kwargs["lr"]
#         del kwargs["lr"]
#
#     return paddle_lr_class(*args, **kwargs)
#
# import sys
# this_module = sys.modules[__name__]
# import inspect
#
# class PaddleLRModuleProxy(object):
#
#     def __getattribute__(self, *args, **kwargs):
#         # Perform custom logic here
#
#         obj = object.__getattribute__(this_module, *args, **kwargs)
#
#         if inspect.isclass(obj) and obj.__module__.startswith("paddle.optimization"):
#             print("LR", obj.__module__)
#             return partial(paddle_lr_class_creator, obj)
#         else:
#             return obj
#
# sys.modules[__name__] = PaddleLRModuleProxy()