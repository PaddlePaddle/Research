#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief cpu_trainer.py
 Author: map(wushilei@baidu.com)
 Date: 2019/07/24 17:53:40
 Brief:
 CPUTrainer for cpu platform training. 
 Please set FLAGS.num_preprocessing_threads for multi-core training.
 User can data_reader as dataset, pyreader, datafeed.
"""

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import logging
import shutil

import paddle.fluid as fluid

from base_frame import BaseFrame
from model_state import VariableState, average_vars, interpolate_vars, dist_get_reduced_vars

class CPUTrainer(BaseFrame):
    """
    CPU datafeed CPUTrainer
    CPUTrainer converts the data that returned by a reader into 
    a data structure that can feed into Executor
    """

    def __init__(self):
        super(CPUTrainer, self).__init__()
        self.ckpt_list = []
    
    def record_checkpoint(self, FLAGS, global_step):
        """
            record checkpoint
            TODO: restore checkpoint, distribute env
        """
        ckpt_path = "%s/checkpoint.meta" % (FLAGS.train_dir)
        path = "%s/checkpoint_%s" % (FLAGS.train_dir, global_step)
        self.ckpt_list.append(path)
        logging.info("save_max_to_keep: %d, current ckpt_list: %s", FLAGS.save_max_to_keep, 
                str(self.ckpt_list))
        if len(self.ckpt_list) > FLAGS.save_max_to_keep:
            ckpt_to_keep = self.ckpt_list[-FLAGS.save_max_to_keep:]
            ckpt_to_remove = set(self.ckpt_list) - set(ckpt_to_keep)
            self.ckpt_list = ckpt_to_keep
            for ckpt in ckpt_to_remove:
                ckpt_dir = ckpt
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)

        ckpt_file = open(ckpt_path, "w")
        i = 1
        ckpt_file_content = "[Monitor]" + "\n" + "ckpt_version: " + str(global_step) + "\n"
        for ckpt in self.ckpt_list:
            if i == len(self.ckpt_list):
                ckpt_str = "init_pretrain_model: " + ckpt + "\n"
                ckpt_str += "init_train_params: None\n"
                ckpt_str += "eval_dir: " + ckpt + "\n"
            else:
                ckpt_str = "ckpt_" + str(i) + ": " + ckpt + "\n"
            i += 1
            ckpt_file_content += ckpt_str
        ckpt_file.write(ckpt_file_content)
        ckpt_file.close()

    def save_model(self, FLAGS, net_output, global_step):
        """
            save model
        """
 
        if global_step != "final" and global_step % FLAGS.save_model_steps != 0:
            return

        path = "%s/checkpoint_%s" % (FLAGS.train_dir, global_step)
        fluid.io.save_inference_model(path, 
                   net_output['model_output']['feeded_var_names'],
                   net_output["model_output"]['fetch_targets'],
                   self.paddle_env['exe'], self.get_infer_program(FLAGS), program_only=True)
        fluid.io.save_persistables(self.paddle_env['exe'], path)
        self.record_checkpoint(FLAGS, global_step)

    def train(self, FLAGS, net_output):
        """
        start training
        """
        program = self.get_main_program(FLAGS)
                
        self.init_model_params(self.paddle_env['exe'], program, FLAGS)
        net_instance = self.paddle_env["factory"]["net"]
 
        if not isinstance(program, fluid.CompiledProgram) and FLAGS.platform == 'local-cpu' \
                and FLAGS.data_reader in ("pyreader", "async"):
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_threads = self.get_thread_num(FLAGS)  #2 for fp32 4 for fp16, CPU_NUM
            exec_strategy.use_experimental_executor = True
            exec_strategy.num_iteration_per_drop_scope = 10  #important shit

            build_strategy = fluid.BuildStrategy()
            build_strategy.remove_unnecessary_lock = False
            build_strategy.enable_inplace = True
            #build_strategy.fuse_broadcast_ops = True
            #build_strategy.memory_optimize = True
            #if exec_strategy.num_threads > 1:
                #build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
            print('-' * 50)
            print('-' * 50)
            program = fluid.CompiledProgram(program).with_data_parallel(
                    loss_name=net_output["loss"].name,
                    build_strategy=build_strategy,
                    exec_strategy=exec_strategy)
        
        # all_var_names = [var.name for var in program.all_parameters()]
        all_var_names = [u'wordid_embedding', u'conv_weight', u'conv_bias', u'prefix_intra_ffn_innerfc_weight', u'prefix_intra_ffn_innerfc_bias', u'prefix_intra_ffn_outerfc_weight', u'prefix_intra_ffn_outerfc_bias', u'prefix_fc_weight', u'prefix_fc_bias', u'loc_fc_weight', u'loc_fc_bias', u'context_fc_weight', u'context_fc_bias', u'cross_inter_ffn_innerfc_weight', u'cross_inter_ffn_innerfc_bias', u'cross_inter_ffn_outerfc_weight', u'cross_inter_ffn_outerfc_bias', u'name_inter_ffn_innerfc_weight', u'name_inter_ffn_innerfc_bias', u'name_inter_ffn_outerfc_weight', u'name_inter_ffn_outerfc_bias', u'addr_inter_ffn_innerfc_weight', u'addr_inter_ffn_innerfc_bias', u'addr_inter_ffn_outerfc_weight', u'addr_inter_ffn_outerfc_bias', u'field_fc_weight', u'field_fc_bias', u'poi_fc_weight', u'poi_fc_bias']
        model_state = VariableState(program, all_var_names)
        global_step = 0
        epoch = 2
        for epoch_id in range(FLAGS.num_epochs_input):
            task_id = 0
            total_loss = 0
            for task in self.paddle_env["data_reader"]():
                #LoD sample
                #define old_vars
                old_vars = model_state.export_variables()
                
                if self.paddle_env["data_feeder"] is not None:
                    task = self.paddle_env["data_feeder"].feed(task)

                new_vars = []
                for _ in range(epoch):
                    for batch_sample in task:
                        result = self.paddle_env["exe"].run(program=program, 
                            feed=batch_sample,
                            fetch_list=self.debug_tensors,
                            return_numpy=False)
                        global_step += 1
                #export var
                new_vars.append(model_state.export_variables())
                model_state.import_variables(old_vars)
                new_vars = average_vars(new_vars)

                if FLAGS.num_gpus == 1:
                    model_state.import_variables(interpolate_vars(old_vars, new_vars, FLAGS.meta_step_size))
                elif FLAGS.num_gpus > 1:
                    #get vars from other workers
                    reduced_vars = dist_get_reduced_vars(self.paddle_env["exe"], new_vars)
                    reduced_vars = [var/float(os.getenv("PADDLE_TRAINERS_NUM")) for var in reduced_vars]
                    model_state.import_variables(interpolate_vars(old_vars, reduced_vars, FLAGS.meta_step_size))

                #import old vars
                # model_state.import_variables(interpolate_vars(old_vars, reduced_vars, FLAGS.meta_step_size))
                #broadcast var from root to workers
                #model_state.broadcast_vars(self.paddle_env["exe"])
                
                net_instance.train_format(result, global_step, epoch_id, task_id)
                task_id += 1
                total_loss += np.mean(np.array(result[0]))
                if FLAGS.max_number_of_steps is not None and global_step >= FLAGS.max_number_of_steps:
                    pass
                
                self.save_model(FLAGS, net_output, global_step)

            logging.info("hxl_print epoch[%d], loss[%f]", epoch_id, total_loss/task_id)
            self.save_model(FLAGS, net_output, global_step)
        
            if FLAGS.max_number_of_steps is not None and global_step >= FLAGS.max_number_of_steps:
                break

        self.save_model(FLAGS, net_output, "final")

    def init_model_params(self, exe, main_program, FLAGS):
        """
            load params of pretrained model, NOT including moment, learning_rate
        """
        if FLAGS.init_train_params is not None:
            place = self.create_places(FLAGS)[0]
            self.paddle_env['factory']['net'].init_params(place)
            logging.info("Load pretrain params from {}.".format(
                 FLAGS.init_train_params))
        elif FLAGS.init_pretrain_model is not None:
            fluid.io.load_persistables(
                exe,
                FLAGS.init_pretrain_model,
                main_program=main_program)
            logging.info("Load pretrain persistables from {}.".format(
                 FLAGS.init_pretrain_model))
        return


if __name__ == '__main__':
    trainer = CPUTrainer()
    ret = trainer.start(sys.argv)
    if not ret:
        sys.exit(-1)
 
