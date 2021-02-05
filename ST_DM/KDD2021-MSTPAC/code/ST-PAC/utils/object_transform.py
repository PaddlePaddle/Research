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
 Specify the brief object_transform.py
 Date: 2019/07/15 16:37:12
"""

import pickle 
import base64 

class ObjectTransform(object):
    """
    transform object and string
    """
    @classmethod
    def pickle_dumps_to_str(cls, obj):
        """
        from object to str
        """
        try:
            #return base64.encodebytes(pickle.dumps(obj)).decode()
            #return base64.b64encode(pickle.dumps(obj))
            return base64.b64encode(pickle.dumps(obj)).decode()
        except pickle.PicklingError:
            pass

    @classmethod
    def pickle_loads_from_str(cls, obj_str):
        """
        from str to object
        """
        try:
            #return pickle.loads(base64.decodebytes(obj_str.encode()))
            #return pickle.loads(base64.b64decode(obj_str))
            return pickle.loads(base64.b64decode(obj_str.encode()))
        except pickle.UnpicklingError:
            pass

