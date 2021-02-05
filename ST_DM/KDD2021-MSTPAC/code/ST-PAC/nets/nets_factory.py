#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################


"""
 Specify the brief datasets_factory.py
 Date: 2019/07/10 17:27:57
"""

models = {}

def register_class(target_class):
    """
    register all nets classs. Including BaseNet and its sub-class
    """
    models[target_class.__name__] = target_class

class Meta(type):
    """
    Meta class, for net class register
    """
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        register_class(cls)
        return cls

def get_model(name):
    """
    get net class type with class name
    """
    return models.get(name, None)

