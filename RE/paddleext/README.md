# Paddle Extension

Paddle extensions, including implementation for torch apis. 

## Install 

* Clone the repo
* Add the path of paddleext folder to PYTHONPATH

## Document

### Seameless shift backend between Paddle and PyTorch

* Add following code to the root __init__.py of your project
(assume your project name is PROJECT):

```python

import importlib
import sys
import os 

BACKEND = os.environ.get('BACKEND', 'paddle')

if BACKEND == "paddle":
 
    from paddleext import torchapi
    sys.modules["PROJECT.backend"] = torchapi

    try:
        import paddlemetrics
        sys.modules["PROJECT.metrics"] = paddlemetrics
    except Exception as e:
        pass

elif BACKEND == "torch":
    try:
        import torch
        import types

        class VirtualModule(types.ModuleType):
            def __init__(self, module_name, sub_modules):

                super().__init__(module_name)
                try:
                    import sys
                    sys.modules[module_name] = self
                    self._module_name = module_name
                    self._sub_modules = sub_modules
                    for sub_name, module in sub_modules.items():
                        if sub_name is None:
                            sys.modules[f"{module_name}"] = module
                        else:
                            sys.modules[f"{module_name}.{sub_name}"] = module
                except ImportError as err:
                    raise err  # please signal error in some useful way :-)

            def __repr__(self):
                return "Virtual module for " + self._module_name

            def __getattr__(self, attrname):

                if attrname in self._sub_modules.keys():
                    import sys
                    return self._sub_modules[attrname]
                else:
                    return super().__getattr__(attrname)


        import pkgutil

        sub_modules = {None: torch}
        for module_info in pkgutil.iter_modules(torch.__path__):
            if not module_info.name.startswith("_"):
                try:
                    module = importlib.import_module("torch." + module_info.name)
                    sub_modules[module_info.name] = module
                except:
                    pass

        VirtualModule("PROJECT.backend", sub_modules)


    except Exception as e:
        raise e

    try:
        import torchmetrics

        sys.modules["PROJECT.metrics"] = torchmetrics
    except Exception as e:
        pass

```
* set the environment variable BACKEND to "paddle" or "torch" to switch backend
* import the backend module in your code

```python
import PROJECT.backend as B
from PROJECT.backend import nn
import PROJECT.metrics as M
```
* replace all "torch." or "paddle." with "B." in your code