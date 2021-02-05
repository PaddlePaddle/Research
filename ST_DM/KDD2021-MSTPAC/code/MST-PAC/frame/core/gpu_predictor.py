#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief gpu_predictor.py
 Author: map(wushilei@baidu.com)
 Date: 2019/07/21 20:09:58

 Brief: 
    This class can cover single-node multi-gpus and multi-node multi-gpus

    The corresponding platform is gpu and slurm.
"""

from __future__ import print_function
import os
import sys
import argparse

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet
#from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
#from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.incubate.fleet.base import role_maker

from utils.object_transform import ObjectTransform
from cpu_predictor import CPUPredictor
from gpu_mixin import GPUMixin

class GPUPredictor(GPUMixin, CPUPredictor):
    """
    gpu trainer class
    This class can cover single-node multi-gpus and multi-node multi-gpus
    The corresponding platform is gpu and slurm.
    """
    def __init__(self):
        super(GPUPredictor, self).__init__()

    def split_filelist(self, FLAGS):
        """
        split filelist for multi-node or multi gpus.
        """
        if self.is_multi_gpu(FLAGS):
            filelist_arr = FLAGS.file_list.split(',')
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
            num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))
            trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
            trainer_endpoints = trainer_endpoints.split(',')

            role = role_maker.UserDefinedCollectiveRoleMaker(current_id=trainer_id,
                    worker_endpoints=trainer_endpoints)
            fleet.init(role)

            filelist_arr = fleet.split_files(filelist_arr) 
            FLAGS.file_list = ','.join(filelist_arr)    


if __name__ == '__main__':
    trainer = GPUPredictor()
    ret = trainer.start(sys.argv)
    if not ret:
        sys.exit(-1)

