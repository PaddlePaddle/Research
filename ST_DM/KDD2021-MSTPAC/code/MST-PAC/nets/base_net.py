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
 Specify the brief base_net.py
 Date: 2019/07/10 15:46:22
"""

import os
import sys
import numpy as np
import logging
import six

from utils.common_lib import CommonLib
from nets.nets_factory import Meta


class RegisterBaseNet(object):
    """
    Base net class 
    """

    def __init__(self, flags):
        self._flags = flags
        self.is_training = (self._flags.dataset_split_name == 'train')
     
    def pred_format(self, result, **kwargs):
        """
        result: one batch predict narray
        format result
        """
        if result is None or result in ['_PRE_', '_POST_']:
            return

        def _flat_str(vec):
            if isinstance(vec, (list, tuple, np.ndarray)): 
                return "%s:%s" % (np.array(vec).shape,
                            ";".join([' '.join(str(j) for j in np.array(i).flatten()) for i in vec]))
            else:
                return str(vec)

        out = '\t'.join([_flat_str(np.array(o)) for o in result])
        #out = '\t'.join(CommonLib.unpack_tensor(result))
        
        print("%s" % out)
    
    def train_format(self, result, global_step, epoch_id, batch_id):
        """
            result: one batch train narray
        """
        if global_step == 0 or global_step % self._flags.log_every_n_steps != 0:
            return
        #result[0] default is loss.
        avg_res = np.mean(np.array(result[0]))
        vec = []
        for i in range(1, len(result)):
            vec += map(str, np.array(result[i]))
        
        logging.info("epoch[%s], global_step[%s], batch_id[%s], extra_info: loss[%s], debug[%s]" % (epoch_id,
                    global_step, batch_id, avg_res, ";".join(vec)))

    def init_params(self, place):
        """
            init pretrain vars vec
            param = fluid.default_main_program().global_block().var('fc.b')
            param.set(xxx, place)
        """
        return 

    def net(self, inputs):
        """
        must implement in subclass
        """
        return 


@six.add_metaclass(Meta)
class BaseNet(RegisterBaseNet):
    """
    base net with metaclass
    """
    pass 

