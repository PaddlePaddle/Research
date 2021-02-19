#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief gpu_trainer.py
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
import logging

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.collective import DistributedStrategy
#from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
#from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.incubate.fleet.base import role_maker

from utils.object_transform import ObjectTransform
from cpu_trainer import CPUTrainer
from gpu_mixin import GPUMixin

class GPUTrainer(GPUMixin, CPUTrainer):
    """
    gpu trainer class
    This class can cover single-node multi-gpus and multi-node multi-gpus
    The corresponding platform is gpu and slurm.
    """
    def __init__(self):
        super(GPUTrainer, self).__init__()
    
    def set_optimizer(self, FLAGS, net_output):
        """
        set optimizer
        """
        optimizer = net_output['optimizer']
        if self.is_multi_gpu(FLAGS):
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
            num_trainers = int(os.getenv("PADDLE_TRAINERS_NUM"))
            trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
            logging.info("train_id:%s, num_trainers:%s, trainer_endpoints:%s" % (trainer_id,
                        num_trainers, trainer_endpoints))
            
            trainer_endpoints = trainer_endpoints.split(',')

            role = role_maker.UserDefinedCollectiveRoleMaker(current_id=trainer_id,
                    worker_endpoints=trainer_endpoints)
            fleet.init(role)
            
            dist_strategy = DistributedStrategy()
            #num_nodes = len(set([x.split(':')[0] for x in trainer_endpoints]))
            #if num_nodes == 1:
            #    dist_strategy.use_local_sgd = True
                #dist_strategy.mode = "collective" #multi node is nccl2
                #dist_strategy.collective_mode = "local_sgd" # local_sgd or grad_allreduce
            #    logging.info("use local sgd, not nccl2 for single node.")

            """
            #TODO:
            dist_strategy.enable_inplace = FLAGS.with_inplace
            if FLAGS.fuse_ops:
                dist_strategy.fuse_all_reduce_ops = 1
            dist_strategy.nccl_comm_num = FLAGS.nccl_comm_num
            """
            optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)

        return optimizer.minimize(net_output['loss'])
  
    def split_filelist(self, FLAGS):
        """
        split filelist for multi-node or multi gpus.
        """
        if self.is_multi_gpu(FLAGS):
            filelist_arr = FLAGS.file_list.split(',') 
            filelist_arr = fleet.split_files(filelist_arr)
            FLAGS.file_list = ','.join(filelist_arr) 
    
    def get_main_program(self, FLAGS):
        """
            train program
        """
        if self.is_multi_gpu(FLAGS): 
            return fleet.main_program
        return super(GPUTrainer, self).get_main_program(FLAGS) 
    
    def get_startup_program(self, FLAGS):
        """
            train program
        """
        if self.is_multi_gpu(FLAGS): 
            return fleet.startup_program
        return super(GPUTrainer, self).get_startup_program(FLAGS) 

    def get_infer_program(self, FLAGS):
        """
            infer program
        """
        if self.is_multi_gpu(FLAGS): 
            return None 
        return super(GPUTrainer, self).get_infer_program(FLAGS) 

    def save_model(self, FLAGS, net_output, global_step):
        """
            save model
        """
        if global_step != "final" and global_step % FLAGS.save_model_steps != 0:
            return

        path = "%s/checkpoint_%s" % (FLAGS.train_dir, global_step)
 
        if self.is_multi_gpu(FLAGS): 
            if fleet.is_first_worker():
                fleet.save_inference_model(self.paddle_env['exe'],
                        path, 
                        net_output['model_output']['feeded_var_names'],
                        net_output['model_output']['fetch_targets'])
                fleet.save_persistables(self.paddle_env['exe'], path)
                self.record_checkpoint(FLAGS, global_step)
        else:
            super(GPUTrainer, self).save_model(FLAGS, net_output, global_step)


if __name__ == '__main__':
    trainer = GPUTrainer()
    ret = trainer.start(sys.argv)
    if not ret:
        sys.exit(-1)

