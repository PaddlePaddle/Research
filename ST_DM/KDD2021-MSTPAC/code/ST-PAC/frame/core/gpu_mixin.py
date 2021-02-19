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
 Specify the brief gpu_mixin.py
 Date: 2019/08/23 09:30:55
 Brief:
    GPUMixin is created for multiple inheritance in both GPUPredictor and GPUTrainer.    
    We can add shared lib functions of GPU operation in this mixin class.
"""

import os
import sys
import argparse

import paddle.fluid as fluid

from base_frame import BaseFrame
from utils.object_transform import ObjectTransform

class GPUMixin(BaseFrame):
    """
    GPUMixin is created for multiple inheritance in both GPUPredictor and GPUTrainer.
    
    We can add shared lib functions of GPU operation in this mixin class.
    """
    def create_places(self, FLAGS):
        """
        create platform places
        fluid.cuda_places(), multi gpu by fleet
        """
        places = None
        if not self.is_multi_gpu(FLAGS):
            places = [fluid.CUDAPlace(0)]
        else:
            gpu_id = 0
            if os.getenv("FLAGS_selected_gpus"):
                gpu_id = int(os.getenv("FLAGS_selected_gpus"))
            places = [fluid.CUDAPlace(gpu_id)]

        return places

    def is_multi_gpu(self, FLAGS):
        """
        is multi gpu or not
        """
        if FLAGS.num_gpus <= 1:
            return False

        return True
  
    def get_thread_num(self, FLAGS):
        """
        get thread num for gpu dataset 
        """
        #if FLAGS.data_reader == "dataset":
        # Get device number:fluid.core.get_cuda_device_count()
        #gpu mode: set thread num as 1
        return 1
        #return super(GPUMixin, self).get_thread_num(FLAGS)

