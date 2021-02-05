#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief cpu_predictor.py
 Author: map(wushilei@baidu.com)
 Date: 2019/08/22 17:06:30
 Brief:
    CPUPredictor is shared for local task and hadoop task with 
    datafeed/pyreader/dataset_factory.

    The hadoop task will get sample from stdin, 
    while local task get sample from file_list
"""

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import logging

import paddle.fluid as fluid

from base_frame import BaseFrame
from utils import flags
from utils.load_conf_file import LoadConfFile

FLAGS = flags.FLAGS

class CPUPredictor(BaseFrame):
    """
    CPUPredictor
    Predict with datareader in local platform.
    """

    def __init__(self):
        super(CPUPredictor, self).__init__()

    def parse_args(self):
        """
            parse args
        """
        self.set_default_args()
        #load user defined conf file
        flags.DEFINE_custom(
            #'conf_file', args.conf_file, 'load flags from conf file', 
            'conf_file', './conf/demo_local.conf', 'load flags from conf file', 
            action=LoadConfFile, sec_name="Evaluate")
        #append additional args  
        self.append_additional_args(FLAGS)

    def run(self, FLAGS):
        """
        start predict 
        """
        #get datasets instance and net instance.
        factory = self.get_factory_instance(FLAGS)
        if not factory:
            return False
 
        
        place = self.create_places(FLAGS)[0]
        exe = fluid.Executor(place)
        net_output = None
        if FLAGS.init_pretrain_model is not None:
            #prepare net 
            self.set_pre_paddle_env(FLAGS, factory)
            net_output = factory['net'].net(factory["inputs"])
            if not self.verify_net_output(net_output, FLAGS):
                return False
            program = self.get_infer_program(FLAGS)
            feed_var_names = net_output['model_output']['feeded_var_names']
            fetch_targets = net_output["model_output"]['fetch_targets']
        else:
            #Load infer model 
            program, feed_var_names, fetch_targets = fluid.io.load_inference_model(
                    "%s" % (FLAGS.eval_dir), exe)
            self.input_names = feed_var_names
            #self.input_layers = [program.global_block().var(name) for name in feed_var_names] 
            self.input_layers = [x for x in self.input_layers if x.name in feed_var_names] 
            self.set_pre_paddle_env(FLAGS, factory)
       
        #prepare paddle environment for predict
        exe.run(self.get_startup_program(FLAGS))
     
        self.init_model_params(exe, program, FLAGS)

        self.paddle_env["exe"] = exe
        self.paddle_env['program'] = program
        self.paddle_env['feeded_var_names'] = feed_var_names
        self.paddle_env['fetch_targets'] = fetch_targets

        self.split_filelist(FLAGS)
        
        self.pred(FLAGS, net_output)
            
        return True 
    
    def init_model_params(self, exe, main_program, FLAGS):
        """
            load params of pretrained model, NOT including moment, learning_rate
        """
        if FLAGS.init_pretrain_model is None:
            return

        fluid.io.load_params(
            exe,
            FLAGS.init_pretrain_model,
            main_program=main_program)

        logging.info("Load pretrain parameters from {}.".format(
             FLAGS.init_pretrain_model))

    def pred(self, FLAGS, net_output):
        """
        run predict with datafeed
        """
        net_instance = self.paddle_env["factory"]["net"]
        #pre process
        net_instance.pred_format('_PRE_', frame_env=self)
        if FLAGS.data_reader == "dataset":
            logging.info("current worker file_list: %s" % FLAGS.file_list)
            self.paddle_env['dataset'].set_filelist(FLAGS.file_list.split(','))
            if FLAGS.num_gpus > 0:
                #gpu mode: set thread num as 1
                self.paddle_env['dataset'].set_thread(1)
     
            if FLAGS.dataset_mode == "InMemoryDataset":
                self.paddle_env['dataset'].load_into_memory()
            fetch_info = [x.name for x in self.paddle_env['fetch_targets']] 
            self.paddle_env['exe'].infer_from_dataset(program=self.paddle_env['program'],
                           dataset=self.paddle_env['dataset'],
                           fetch_list=self.paddle_env['fetch_targets'],
                           fetch_info=fetch_info,
                           print_period=1)
            #FIXME: dataset not support batch level step
            #net_instance.pred_format(None)
        elif (FLAGS.data_reader == "async" or FLAGS.data_reader == "pyreader") \
            and not FLAGS.py_reader_iterable:
            prog = self.paddle_env['program']
            #exe_strategy = fluid.ExecutionStrategy()
            ## to clear tensor array after each iteration
            #exe_strategy.num_iteration_per_drop_scope = 1
            #prog = fluid.CompiledProgram(prog).with_data_parallel(
            #    exec_strategy=exe_strategy, places=self.create_places(FLAGS)[0])

            #Before the start of every epoch, call start() to invoke PyReader;
            self.paddle_env["data_reader"].start()
            try:
                while True:
                    result = self.paddle_env["exe"].run(program=prog,
                                     fetch_list=self.paddle_env['fetch_targets'],
                                     return_numpy=False)
                    net_instance.pred_format(result)
            except fluid.core.EOFException:
                """
                At the end of every epoch, read_file throws exception 
                fluid.core.EOFException . Call reset() after catching up 
                exception to reset the state of PyReader in order to start next epoch.
                """
                self.paddle_env["data_reader"].reset() 
        else:
            for sample in self.paddle_env["data_reader"]():
                if self.paddle_env["data_feeder"] is not None:
                    sample = self.paddle_env["data_feeder"].feed(sample) 

                result = self.paddle_env["exe"].run(program=self.paddle_env['program'], 
                            feed=sample,
                            fetch_list=self.paddle_env['fetch_targets'],
                            return_numpy=False)
                net_instance.pred_format(result)
        #post process
        net_instance.pred_format('_POST_', frame_env=self)

        
if __name__ == '__main__':
    predictor = CPUPredictor()
    ret = predictor.start(sys.argv)
    if not ret:
        sys.exit(-1)
 
