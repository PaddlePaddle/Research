#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief gpu_mixin.py
 Author: map(wushilei@baidu.com)
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

