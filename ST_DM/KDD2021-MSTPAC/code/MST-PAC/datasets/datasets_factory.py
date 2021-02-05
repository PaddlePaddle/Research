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
 Brief:
    Management of all dataset sub-classes. 
    Frame can get all user-defined dataset class from this factroy.
"""

class Meta(type):
    """
    Meta class , for subclass register
    """
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        DatasetsFactory.register_class(cls)
        return cls

class DatasetsFactory(object):
    """
    DatasetsFactory: all datasets class and objects
    """
    datasets = {}
    datasets_instances = {}

    @classmethod
    def register_class(cls, target_class):
        """
        register for all dataset class.
        including BaseDataset and its subclass
        """
        cls.datasets[target_class.__name__] = target_class

    @classmethod
    def get_dataset(cls, name):
        """
        get class type with class name
        """
        return cls.datasets.get(name, None)
   
    @classmethod
    def set_instance(cls, name, instance=None):
        """
        store datasets class objects
        """
        if not instance:
            return None
        cls.datasets_instances[name] = instance
   
    @classmethod
    def get_instance(cls, name, instance=None):
        """
        get dataset object with className
        """
        if name in cls.datasets_instances:
            instance = cls.datasets_instances[name]
        else:
            if not instance:
                return None
            cls.datasets_instances[name] = instance
        return instance

