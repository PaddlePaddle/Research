#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import distutils.util
import numpy as np
import six
from collections import deque
import argparse
import functools
from edict import AttrDict
import pdb

_C = AttrDict()
cfg = _C

#
# Training options
#

_C.data_dir = './veri'
# Snapshot period
_C.snapshot_iter = 2000

_C.num_instances = 1

_C.batch_size = 64

# pixel mean values
_C.pixel_means = [0.485, 0.456, 0.406]

# pixel std values
_C.pixel_stds = [0.229, 0.224, 0.225]


# derived learning rate the to get the final learning rate.
_C.learning_rate = 0.001

# maximum number of iterations
_C.max_iter = 100000


#_C.warm_up_iter = 4000
_C.warm_up_iter = 100
_C.warm_up_factor = 0.


_C.lr_steps = [10000, 16000, 20000]
#_C.lr_steps = [20000, 32000, 40000]
#_C.lr_steps = [200000, 320000, 400000]
#_C.lr_steps = [100000, 160000, 200000]
_C.lr_gamma = 0.1

# L2 regularization hyperparameter
_C.weight_decay = 0.0005

# momentum with SGD
_C.momentum = 0.9


# support both CPU and GPU
_C.use_gpu = True

_C.class_num = 751

def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def print_arguments_dict(cfgs):

    
    print("-----------  Configuration Arguments -----------")
    for key, value in cfgs.items():
        print('%s: %s' % (key, value))
    print("------------------------------------------------")
def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self):
        self.loss_sum = 0.0
        self.iter_cnt = 0

    def add_value(self, value):
        self.loss_sum += np.mean(value)
        self.iter_cnt += 1

    def get_mean_value(self):
        return self.loss_sum / self.iter_cnt


def merge_cfg_from_args(args):
    """Merge config keys, values in args into the global config."""
    for k, v in sorted(six.iteritems(vars(args))):
        try:
            value = eval(v)
        except:
            value = v
        _C[k] = value


def parse_args():
    """return all args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    # ENV
    add_arg('use_gpu',          bool,   True,                                  "Whether use GPU.")
    add_arg('model_arch',       str,    'ResNet101_vd',                        "The model arch to train.")

    add_arg('pretrain',         str,    'pretrained/ResNet101_vd_pretrained',  "The pretrain model path.")
    add_arg('weights',          str,    'model_iter129999',                    "The weights path.")

    add_arg('data_dir',         str,    'dataset/aicity20_all',                "The data root path.")
    add_arg('model_save_dir',   str,    'output',                              "The path to save model.")

    #SOLVER
    add_arg('batch_size',       int,    64,     "Mini-batch size per device.")
    add_arg('test_batch_size',  int,    32,     "Mini-batch size per device.")
    add_arg('num_instances',    int,    4,      "Mini-batch size per device.")
    add_arg('learning_rate',    float,  0.01,   "Learning rate.")
    add_arg('warm_up_iter',     float,  8000,   "Learning rate.")
    add_arg('start_iter',       int,    0,      "Start iteration.")
    add_arg('max_iter',         int,    230000, "Iter number.")    
    add_arg('snapshot_iter',    int,    3000,   "Save model every snapshot stride.")
    add_arg('lr_steps', nargs='+', type=int, default=[100000, 160000, 200000], help="The mean of input image data")


    add_arg('margin',     float,  0.3,  "intra class margin for TripletLoss.")


    # TRAIN TEST INFER
    add_arg('big_height',       int,    384,    "Image big_height.")
    add_arg('big_width',        int,    384,    "Image big_width.")
    add_arg('target_height',    int,    384,    "Image target_height.")
    add_arg('target_width',     int,    384,    "Image target_width.")
    
    add_arg('padding_size',     int,    10,     "Image padding size.")
    add_arg('re_prob',          float,  0.5,    "Image random erase probility.")

    add_arg('use_flip',         bool,   False,  "Whether use flip.")
    add_arg('flip_test',        bool,   False,  "Whether use flip in test.")
    add_arg('use_autoaug',      bool,   False,  "Whether use autoaug.")
    add_arg('use_crop',         bool,   False,  "Whether use crop.")


    add_arg('use_multi_branch', bool,   False,   "whether using multi_branch_arch")
    add_arg('num_features',     int,    512,    "feature dims.")
    add_arg('syncbn',           bool,   True,   "Whether to use synchronized batch normalization.")


    args = parser.parse_args()
    merge_cfg_from_args(args)
    return cfg


