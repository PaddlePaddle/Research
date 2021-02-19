#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief paddlecloud_fleet_trainer.py
 Author: map(wushilei@baidu.com)
 Date: 2019/08/06 15:00:23
 Brief:
    Training on paddlecloud with fleet.
"""

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import logging

import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
#from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
#from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.base import role_maker
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

from cpu_trainer import CPUTrainer
from utils.common_lib import CommonLib
from utils.object_transform import ObjectTransform


class PaddleCloudFleetTrainer(CPUTrainer):
    """
    local trainer class
    """
    def __init__(self):
        super(PaddleCloudFleetTrainer, self).__init__()
    
    def set_optimizer(self, FLAGS, net_output):
        """
        set optimizer
        """
        optimizer = net_output['optimizer']
        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = (FLAGS.data_reader != "dataset")
        #pslib, strategy = {"use_cvm": True}
 
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        return optimizer.minimize(net_output['loss'])
    
    def split_filelist(self, FLAGS):
        """
        split filelist
        paddlecloud split auto
        """
        if FLAGS.platform == 'pserver-local':
            filelist_arr = FLAGS.file_list.split(',') 
            filelist_arr = fleet.split_files(filelist_arr)
            FLAGS.file_list = ','.join(filelist_arr)
        return 

    def is_server(self):
        """
            set default role of current node
        """
        return fleet.is_server() 

    def run_server(self, FLAGS):
        """
        set default run_server
        """
        #TODO: load pre model
        fleet.init_server(FLAGS.init_pretrain_model)
        if FLAGS.init_train_params is not None:
            place = fluid.CPUPlace()
            self.paddle_env['factory']['net'].init_params(place)
        logging.info("PServer init success!")
        fleet.run_server() 
        
        return True

    def append_additional_args(self, FLAGS):
        """
        append addtional args from the existing args
        """
        #dataset_dir and train_dir is defined in padllecloud, cannot be set by user
        role = role_maker.PaddleCloudRoleMaker() 
        fleet.init(role)

        return super(PaddleCloudFleetTrainer, self).append_additional_args(FLAGS)
    
    def get_main_program(self, FLAGS):
        """
            train program
        """
        return fleet.main_program

    def get_startup_program(self, FLAGS):
        """
            startup program
        """
        return fleet.startup_program

    def get_infer_program(self, FLAGS):
        """
            infer program
        """
        return None 
    
    def save_model(self, FLAGS, net_output, global_step):
        """
            save model
        """
        if (global_step != "final" and global_step % FLAGS.save_model_steps != 0) \
                or not fleet.is_first_worker():
            return

        path = "%s/checkpoint_%s" % (FLAGS.train_dir, global_step)
        fleet.save_inference_model(self.paddle_env['exe'],
                path, 
                net_output['model_output']['feeded_var_names'],
                net_output['model_output']['fetch_targets'])
        #or
        fleet.save_persistables(self.paddle_env['exe'], path)
        self.record_checkpoint(FLAGS, global_step)

    def init_model_params(self, exe, main_program, FLAGS):
        """ 
            init model params in run_server: pserver
            #worker no need init params
        """
        return

    def train(self, FLAGS, net_output):
        """
        start training
        """
        fleet.init_worker()
        super(PaddleCloudFleetTrainer, self).train(FLAGS, net_output)
        fleet.stop_worker()


if __name__ == '__main__':
    trainer = PaddleCloudFleetTrainer()
    ret = trainer.start(sys.argv)
    if not ret:
        sys.exit(-1)


