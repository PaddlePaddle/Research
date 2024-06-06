from functools import partial

from paddle.optimizer import *

from . import lr_scheduler

# import sys
# this_module = sys.modules[__name__]
# import inspect
#
# def paddle_optim_class_creator(paddle_optim_class, *args, **kwargs):
#     """
#
#     Args:
#         paddle_optim_class:
#         *args:
#         **kwargs:
#
#     Returns:
#
#     """
#     if "params" in kwargs:
#         kwargs["parameters"] = kwargs["params"]
#         del kwargs["params"]
#     if "lr" in kwargs:
#         kwargs["learning_rate"] = kwargs["lr"]
#         del kwargs["lr"]
#
#     return paddle_optim_class(*args, **kwargs)
#
# from . import lr
#
# class PaddleOptimModuleProxy(object):
#
#     def __getattribute__(self, *args, **kwargs):
#         # Perform custom logic here
#
#         obj = object.__getattribute__(this_module, *args, **kwargs)
#
#         if inspect.isclass(obj) and obj.__module__.startswith("paddle.optimization"):
#             print(obj.__module__)
#             return partial(paddle_optim_class_creator, obj)
#         else:
#             return obj
#
#
#
# sys.modules[__name__] = PaddleOptimModuleProxy()