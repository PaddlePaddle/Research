# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse


def args_parser():
    '''Common classifier application command-line arguments.
    '''
    parser = argparse.ArgumentParser(
        description='image classification model command-line')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DenseNet121')
    parser.add_argument('--data', '-d', default='./data')
    parser.add_argument('--resume-from', dest='resumed_checkpoint_path', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint. Use to resume paused training session.')
    parser.add_argument('--infer-file', dest='infer_file')
    parser.add_argument('--infer-classdim', dest='infer_classdim', default=5)

    return parser
