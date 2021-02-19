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
 Specify the brief base_frame.py
 Brief:
    BaseFrame is the core basic class of our frame.
    It is ancestor-class for all predictor and trainer. 
"""

from __future__ import print_function
import os
import sys
import argparse
from collections import OrderedDict
import numpy as np
import random
import logging
import paddle
import paddle.fluid as fluid

from datasets import datasets_factory
from nets import nets_factory
from utils import flags
from utils import logger
from utils.load_conf_file import LoadConfFile
from sample_reader import SampleReader
from utils.object_transform import ObjectTransform
from utils.common_lib import CommonLib

FLAGS = flags.FLAGS


class BaseFrame(object):
    """
    Base Trainer: Define shared method for sub-classes
    """
    def __init__(self):
        """
        init member
        """
        self.input_layers = []
        self.input_names = []
        self.debug_tensors = []
        self.debug_keys = []
        self.paddle_env = {}

    def parse_args(self):
        """
        parse args and load config from conf file
        """
        #init ArgumentParser
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--conf_file")
        args = parser.parse_args()
        """
        
        """
        set the necessary default args. 
        Fault-tolerant in frame, even though
        the necessary args is not define in user conf file.
        """
        self.set_default_args()
        #load user defined conf file
        flags.DEFINE_custom(
            #'conf_file', args.conf_file, 'load flags from conf file', 
            'conf_file', './conf/demo_local.conf', 'load flags from conf file', 
            action=LoadConfFile, sec_name="Train")

        #append additional args  
        self.append_additional_args(FLAGS)

        if FLAGS.debug_mode: 
            logging.info("base_lr: %f\n"
                   "CUDA_VISIBLE_DEVICES: %s\n"
                   "num_gpus: %d\n"
                   "file_list: %s\n"
                   "dataset_dir: %s\n"
                   %
                   (FLAGS.base_lr,
                    str(FLAGS.cuda_visible_devices),
                    FLAGS.num_gpus,
                    FLAGS.file_list,
                    FLAGS.dataset_dir
                   )
                 )
        return
 
    def set_default_args(self):
        """
        set default flags.
        These default flags will work when user doesnot define in conf file.
        These default flags will be covered when user has defined in conf file.
        """
        logger.init_log("./logs/paddle_frame")
        
        flags.DEFINE_string('dataset_dir', './train_data/', 'set default dataset_dir')

        flags.DEFINE_string('file_list', None, 'set default file_list')

        flags.DEFINE_string('file_pattern', 'part-', 'set sample filename pattern')

        flags.DEFINE_integer('batch_size', 1024, 'set default batch_size')

        flags.DEFINE_string('data_reader', 'pyreader', 'set default data_reader')
        
        flags.DEFINE_string('dataset_split_name', 'train', 'set default dataset_split_name')

        flags.DEFINE_string('dataset_mode', 'QueueDataset', 'set default dataset_mode')

        flags.DEFINE_integer('sample_seed', 1234, 'set default seed')
        
        flags.DEFINE_integer('num_gpus', 0, 'set default gpu index')

        flags.DEFINE_boolean('debug_mode', False, 'set default debug model')

        flags.DEFINE_string('platform', 'local-cpu', 'set default platform.')
        
        flags.DEFINE_string('init_pretrain_model', None, 'set init pretrain model with same network')
        
        flags.DEFINE_string('init_train_params', None, 'set init model params for train, e.g. glue word2vec.')

        flags.DEFINE_integer('num_epochs_input', 2, 'set default epochs')
        
        flags.DEFINE_integer('num_samples', 100, 'set default samples num')
        
        flags.DEFINE_integer('max_number_of_steps', None, 'set default max step num')
        
        flags.DEFINE_float('base_lr', 0.01, 'set default learning rate')

        flags.DEFINE_integer('py_reader_capacity', 128, 'set default py_reader capacity.')

        flags.DEFINE_boolean('py_reader_use_double_buffer', True, 
                             'set_default py_reader use_double_buffer')

        flags.DEFINE_boolean('py_reader_iterable', True, 'set_default py_reader iterable')

        flags.DEFINE_integer('batch_shuffle_size', 0, 'batch data shuffle size, 0 not shuffle')
        
        flags.DEFINE_integer('num_preprocessing_threads', 1, 'num_preprocessing_threads for sample read')
        
        flags.DEFINE_integer('save_model_steps', 100, 'save model in steps')
        
        flags.DEFINE_boolean('reader_batch', False, 'read batch from user dataset')
        
        flags.DEFINE_boolean('drop_last_batch', True, 'drop last batch')
        
        flags.DEFINE_boolean('use_fp16', False, 'fp16')
        
        flags.DEFINE_float('init_loss_scaling', 1.0, 'init_loss_scaling')
        
        flags.DEFINE_integer('incr_every_n_steps', 1000, 'incr_every_n_steps')
        
        flags.DEFINE_integer('decr_every_n_nan_or_inf', 2, 'fp16 decr_every_n_nan_or_inf')
        
        flags.DEFINE_float('incr_ratio', 2.0, 'fp16 incr_ratio')
        
        flags.DEFINE_float('decr_ratio', 0.8, 'fp16 decr_ratio')
        
        flags.DEFINE_boolean('use_dynamic_loss_scaling', True, 'dynamic_loss_scaling')
        
    def append_additional_args(self, FLAGS):
        """
        append addtional args from the existing args
        """ 
        if FLAGS.file_list is None and os.path.exists(FLAGS.dataset_dir):
            file_list = ",".join([FLAGS.dataset_dir.strip() + "/%s" % 
                        x for x in os.listdir(FLAGS.dataset_dir) 
                        if x.startswith(FLAGS.file_pattern)])
            FLAGS.file_list = file_list
        
        if FLAGS.dataset_mode == "Memory":
            FLAGS.dataset_mode = "InMemoryDataset"
        elif FLAGS.dataset_mode == "Queue":
            FLAGS.dataset_mode = "QueueDataset"

        if FLAGS.file_list is None or len(FLAGS.file_list.strip()) < 1:
            #NOTE: No datasetdir on paddlecloud pserver 
            logging.error("dataset not exists [%s]" % FLAGS.dataset_dir)
            return False
        
        return True 

    def verify_net_input(self, context, FLAGS):
        """
        verify the output of parse_contex
        """
        if not context.get("inputs"):
            logging.info("Please set inputs for output in %s" % FLAGS.model_name)
            return False
        
        #check and reset "inputs"
        diff = False 
        for key, value in context["inputs"].items():
            if key != value.name:
                diff = True
                break
        if diff:
            frame_inputs = OrderedDict()
            for value in context["inputs"].values():
                frame_inputs[value.name] = value
            context["inputs"] = frame_inputs

        if not context.get("debug_list"):
            context['debug_list'] = [] 
           
        return True 

    def verify_net_output(self, net_output, FLAGS):
        """
        verify net output
        """
        if not net_output.get("model_output"):
            logging.info("Get model_output from net_output failure.")
            return False
        
        #if not train, return
        if FLAGS.dataset_split_name != 'train':
            return True

        if net_output["model_output"].get("loss"):
            net_output["loss"] = net_output["model_output"]["loss"]

        if not net_output.get("loss"):
            logging.info("Get loss from net_output failure.")
            return False
        
        #TODO support user defined optimizer
        if 'optimizer' not in net_output:
            #default
            optimizer = fluid.optimizer.AdamOptimizer(
                FLAGS.base_lr, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2,
                epsilon=FLAGS.opt_epsilon)
            #optimizer = fluid.optimizer.SGD(learning_rate=FLAGS.base_lr)
            net_output['optimizer'] = optimizer

        if FLAGS.use_fp16:
            net_output['optimizer'] = fluid.contrib.mixed_precision.decorate(
                optimizer=net_output['optimizer'], init_loss_scaling=FLAGS.init_loss_scaling,
                incr_every_n_steps=FLAGS.incr_every_n_steps,
                decr_every_n_nan_or_inf=FLAGS.decr_every_n_nan_or_inf,
                incr_ratio=FLAGS.incr_ratio,
                decr_ratio=FLAGS.decr_ratio,
                use_dynamic_loss_scaling=FLAGS.use_dynamic_loss_scaling)

        #make loss first key
        self.debug_keys.insert(0, "loss")
        self.debug_tensors.insert(0, net_output.get("loss"))

        for k in net_output["debug_output"]:
            if k not in self.debug_keys:
                self.debug_keys.append(k)
                self.debug_tensors.append(net_output["debug_output"][k])

        logging.info("train debug tensors:%s" % self.debug_keys)
        return True
           
    def get_dataset_instance(self, FLAGS):
        """
        get dataset instance from dataset factory
        """
        DatasetClass = datasets_factory.DatasetsFactory.get_dataset(FLAGS.dataset_name)
        if not DatasetClass:
            logging.info("Get DatasetClass failure. "
                  "Invalid dataset name in config: %s" % FLAGS.dataset_name)
            return None
    
        dataset_instance = DatasetClass(FLAGS)
         
        """
        get dataset context.
        """
        inputs = OrderedDict()
        context = dataset_instance.parse_context(inputs)
 
        if not self.verify_net_input(context, FLAGS):
            return None
 
        self.input_names = [key for key in context["inputs"]]
        self.input_layers = [context["inputs"][key] for key in self.input_names]

        self.debug_keys = context["debug_list"]
        self.debug_tensors = [value for value in self.input_layers if value.name in self.debug_keys]

        return dataset_instance, context

    def get_net_instance(self, FLAGS):
        """
        get net instance from net factory
        """
        NetClass = nets_factory.get_model(FLAGS.model_name)
        if not NetClass:
            logging.info("Get NetClass failure."
                  "Invalid model name in config: %s" % FLAGS.model_name)
            return None
    
        net_instance = NetClass(FLAGS)
        return net_instance

    def get_factory_instance(self, FLAGS):
        """
        get dataset and net from the corresponding factory
        """
        factory = {}
        dataset_instance, context = self.get_dataset_instance(FLAGS)

        factory["dataset"] = dataset_instance
        factory["inputs"] = context["inputs"] 
        
        net_instance = self.get_net_instance(FLAGS)
        factory["net"] = net_instance
    
        return factory
 
    def set_optimizer(self, FLAGS, net_output):
        """
        set optimizer
        """ 
        optimizer = net_output['optimizer']
        return optimizer.minimize(net_output['loss'])

    def set_pre_paddle_env(self, FLAGS, factory):
        """
        set paddle env before nets 
        """
        data_reader = self.create_data_reader(FLAGS, factory)
        self.paddle_env["data_reader"] = data_reader

        data_feeder = self.create_data_feeder(FLAGS, factory)
        self.paddle_env["data_feeder"] = data_feeder
        
        dataset = self.create_dataset(FLAGS, factory) 
        self.paddle_env["dataset"] = dataset
        
        self.paddle_env["factory"] = factory

        return True

    def import_user_modules(self, FLAGS):
        """
        import user modules
        """
        if not FLAGS.import_user_modules:  
            return False

        modules = FLAGS.import_user_modules.split(',')
        for module in modules:
            #logging.info("module is %s" % module)
            if not module.strip():
                continue
            __import__(module)

        return True

    def is_server(self):
        """
        set default role of current node
        """
        return False

    def run_server(self, FLAGS):
        """
        set default run_server
        """
        return

    def run_worker(self, FLAGS, net_output):
        """
        set default run worker
        """
        #start training
        self.train(FLAGS, net_output)
        
        return True

    def run(self, FLAGS):
        """
            run frame, default train
        """

        #get datasets instance and net instance.
        factory = self.get_factory_instance(FLAGS)
        if not factory:
            return False

        self.set_pre_paddle_env(FLAGS, factory)

        #prepare net 
        # print (factory["inputs"])
        net_output = factory['net'].net(factory["inputs"])
        if not self.verify_net_output(net_output, FLAGS):
            return False
    
        self.get_infer_program(FLAGS)
    
        if 'optimizer_weight_decay_fn' in net_output:
            param_dict = {} #save raw param
            for param in fluid.default_main_program().global_block().all_parameters():
                param_dict[param.name] = param * 1.0
                param_dict[param.name].stop_gradient = True

        _, param_grads = self.set_optimizer(FLAGS, net_output)
        
        #mem_info = fluid.contrib.memory_usage(program=fluid.default_main_program(),
        #        batch_size=FLAGS.batch_size)
        #logging.info("theoretical memory usage: %s", mem_info)

        if 'optimizer_weight_decay_fn' in net_output:
            for param, grad in param_grads:
                net_output['optimizer_weight_decay_fn'](param, grad, param_dict)

        if self.is_server():
            self.run_server(FLAGS)
        else:
            #prepare paddle environment
            self.set_post_paddle_env(FLAGS, factory)
            self.run_worker(FLAGS, net_output)

        logging.info("paddle training stopped.")
        return True

    def start(self, argv):
        """
        start 
        """
        #parse commandline arguments and conf file.
        self.parse_args()
    
        import_ret = self.import_user_modules(FLAGS)
        if not import_ret:
            logging.info("Import user modules failure.")
            return False
        
        if FLAGS.init_train_params is not None and FLAGS.init_pretrain_model is not None:
            logging.error("init_train_params and init_pretrain_model cannot be both set, conflict!")
            return False

        if FLAGS.reader_batch and FLAGS.data_reader not in ('pyreader', 'async'):
            logging.error("reader_batch only support pyreader")
            return False

        if os.environ.get('FLAGS_eager_delete_tensor_gb', None) is None:
            os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
       
        if os.environ.get('FLAGS_sync_nccl_allreduce', None) is None:
            os.environ["FLAGS_sync_nccl_allreduce"] = "1" 
        #default:multiprocessing.cpu_count()
        os.environ["CPU_NUM"] = str(FLAGS.num_preprocessing_threads)
        
        """
        Print Arguments
        """
        logging.info('-----------  Configuration Arguments -----------')
        for arg, value in sorted(flags.get_flags_dict().items()):
            logging.info('%s: %s' % (arg, value))
        logging.info('------------------------------------------------')

        fluid.default_startup_program().random_seed = FLAGS.sample_seed
        fluid.default_main_program().random_seed = FLAGS.sample_seed
        np.random.seed(FLAGS.sample_seed)
        random.seed(FLAGS.sample_seed)
        return self.run(FLAGS)
    
    def create_places(self, FLAGS):
        """
        create platform places
        fluid.cpu_places()
        """
        places = [fluid.CPUPlace()]
        return places

    def get_startup_program(self, FLAGS):
        """
        get startup program.
        """
        #user can redefined startup program
        startup_program = fluid.default_startup_program()
        return startup_program

    def get_main_program(self, FLAGS):
        """
        get main program
        """
        #user can redefined main program
        main_program = fluid.default_main_program()
        return main_program
    
    def get_infer_program(self, FLAGS):
        """
        get infer program
        """
        infer_program = fluid.default_main_program().clone(for_test=True)
        return infer_program
   
    def init_model_params(self, exe, main_program, FLAGS):
        """
            init pretrain model
        """
        return
        
    def create_executor(self, FLAGS):
        """
        create executor for the specified platform
        """
        place = self.create_places(FLAGS)[0]
        program = self.get_startup_program(FLAGS)

        exe = fluid.Executor(place)
        exe.run(program) 
         
        #TODO parallel executor
        #parallel_exe = fluid.ParallelExecutor(
        #    use_cuda=FLAGS.use_cuda,
        #    loss_name=avg_cost.name,
        #    main_program=self.get_main_program(FLAGS))
        #device_count = parallel_exe.device_count
        #logging.info("device count: %d" % device_count)
        
        return exe 

    def get_thread_num(self, FLAGS):
        """
        get thread_num for multi_thread dataset.
        """
        return FLAGS.num_preprocessing_threads

    def create_dataset(self, FLAGS, factory):
        """
        DatasetFactory is a factory which create dataset by its name, 
        We can create "QueueDataset" or "InMemoryDataset",  or 
        "FileInstantDataset" the default is "QueueDataset". 
        """
        if FLAGS.data_reader != "dataset":
            return None

        dataset = fluid.DatasetFactory().create_dataset(FLAGS.dataset_mode)
        dataset.set_batch_size(FLAGS.batch_size) 
        dataset.set_use_var(self.input_layers)

        dir_name = os.path.dirname(__file__)
        pipe_command = (FLAGS.fluid_bin + " " + dir_name + "/dataset_reader.py " + 
                        ObjectTransform.pickle_dumps_to_str(factory['dataset']) + " " + 
                        ObjectTransform.pickle_dumps_to_str(self.input_names))
   
        """
        Set pipe command of current dataset 
        A pipe command is a UNIX pipeline command
        """
        dataset.set_pipe_command(pipe_command)
        #TODO: shuffle
        #Set thread num, it is the num of readers.
        dataset.set_thread(self.get_thread_num(FLAGS)) 

        return dataset 

    def create_data_reader(self, FLAGS, factory):
        """
        create data_reader object
        """  
        if FLAGS.data_reader == "dataset":
            return None
        sample_reader = SampleReader.get_sample_reader(factory['dataset'], self.input_names)

        if not FLAGS.reader_batch:
            sample_reader = paddle.batch(sample_reader, batch_size=FLAGS.batch_size)
        #sample_reader = paddle.reader.buffered(sample_reader, FLAGS.batch_size * self.get_thread_num(FLAGS))
        #shuffle or not
        if FLAGS.batch_shuffle_size > 0:
            sample_reader = paddle.reader.shuffle(sample_reader,
                    buf_size=FLAGS.batch_shuffle_size)

        if FLAGS.data_reader == "pyreader" or FLAGS.data_reader == "async":
            #py_reader = fluid.io.DataLoader.from_generator(feed_list=self.input_layers, 
            py_reader = fluid.io.PyReader(feed_list=self.input_layers, 
                                          capacity=FLAGS.py_reader_capacity, 
                                          use_double_buffer=FLAGS.py_reader_use_double_buffer,
                                          iterable=FLAGS.py_reader_iterable)
            places = None 
            if FLAGS.py_reader_iterable:
                places = self.create_places(FLAGS)
                if FLAGS.dataset_split_name == 'train' and FLAGS.platform == 'local-cpu' \
                        and isinstance(places[0], fluid.CPUPlace):
                    places = places * self.get_thread_num(FLAGS)

            if FLAGS.reader_batch:
                #py_reader.set_batch_generator(sample_reader, places)
                py_reader.decorate_batch_generator(sample_reader, places)
            else:
                #py_reader.set_sample_list_generator(sample_reader, places)
                py_reader.decorate_sample_list_generator(sample_reader, places)

            return py_reader

        return sample_reader

    def create_data_feeder(self, FLAGS, factory):
        """
        create data_feeder.
        The DataFeed class converts data types such as numpy array into a 
        LoDTensor type to feed the training/inference network
        """
        if FLAGS.data_reader == "dataset" or (FLAGS.data_reader == "pyreader" \
            or FLAGS.data_reader == "async"): 
            return None

        place = self.create_places(FLAGS)[0]
        feeder = fluid.DataFeeder(feed_list=self.input_layers, place=place)
        return feeder
    
    def split_filelist(self, FLAGS):
        """
        split filelist for multi-node or multi gpus.
        """
        return None

    def set_post_paddle_env(self, FLAGS, factory):
        """
        set paddle env. eg. dataset, exe, .
        """
        #important, distribute worker need part of files
        self.split_filelist(FLAGS)
        
        if FLAGS.data_reader == "dataset":
            logging.info("current worker file_list: %s" % FLAGS.file_list)
            self.paddle_env['dataset'].set_filelist(FLAGS.file_list.split(','))

            if FLAGS.dataset_mode == "InMemoryDataset":
                self.paddle_env['dataset'].load_into_memory()
                #TODO: batch_shuffle

        exe = self.create_executor(FLAGS)
        self.paddle_env["exe"] = exe

        return True
  
