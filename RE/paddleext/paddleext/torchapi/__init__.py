import inspect

from .core import *
from .tensor_ import *
from .functional import *
from . import sampler
from . import data
from . import nn
from . import distributed
from . import cuda
from . import optim

#from . import paddle_func

this_module = sys.modules[__name__]


def get_module_attribute(module, *args, **kwargs):
    # Perform custom logic here

    obj = object.__getattribute__(module, *args, **kwargs)

    print("input module:", module)
    print("result object", obj)
    if isinstance(obj, types.FunctionType):
        if not obj.__module__.startswith("paddleext.torchapi."):
            return partial(paddle_delegate_func, obj)
        else:
            return obj
    elif isinstance(obj, types.ModuleType):
        print("result module: " + obj.__name__)
        return ModuleDelegate(obj)
    elif inspect.isclass(obj):
        print("result class: " + obj.__name__)
        return obj
    else:
        return obj

class ModuleDelegate(object):
    def __init__(self, module):
        self.module = module

    def __getattribute__(self, *args, **kwargs):

        module = object.__getattribute__(self, "module")
        result = object.__getattribute__(module, *args, **kwargs)
        if isinstance(result, types.ModuleType):
            return ModuleDelegate(result)
        elif isinstance(result, types.FunctionType):
            if not result.__module__.startswith("paddleext.torchapi."):
                return partial(paddle_delegate_func, result)
            else:
                return result
        elif inspect.isclass(result):
            if result.__module__.startswith("paddle."):
                return make_delegate_class(result)
            else:
                return result
        else:
            return result


    # def __getattr__(self, *args, **kwargs):
    #     return get_module_attribute(self.module, *args, **kwargs),

    # def __delattr__(self, *args, **kwargs):
    #     return object.__delattr__(self.module, *args, **kwargs)
    #
    # def __dir__(self):
    #     return dir(self.module)



sys.modules[__name__] = ModuleDelegate(sys.modules[__name__])
