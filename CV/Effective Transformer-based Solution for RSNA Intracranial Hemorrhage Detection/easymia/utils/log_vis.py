# -*-coding utf-8 -*-
##########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
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
#
##########################################################################
"""
训练日志可视化
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class LogVisual(object):
    """
    读取paddlepaddle日志文件，并进行可视化展示
    """

    def __init__(self, log_dir, save_iter=200):
        """
        initialization
        """
        self.log_dir = log_dir
        self.save_iter = save_iter

    def read_vis(self):
        """
        读取log文件
        """
        log_file_path = os.path.join(self.log_dir, 'workerlog.0')

        f_log = open(log_file_path, 'r')

        epoch_arr = []
        loss_arr = []
        lr_arr = []

        eval_arr_g = []
        eval_arr_p = []

        for line in f_log.readlines():
            if '[TRAIN] Iter:' in line:
                epoch = int(line.split('Iter: ')[1].split(', loss: ')[0].split('/')[0])
                loss = float(line.split('Iter: ')[1].split(', loss: ')[1].split(', lr: ')[0])
                lr = float(line.split('Iter: ')[1].split(', loss: ')[1].split(', lr: ')[1].split(', batch_cost:')[0])

                epoch_arr.append(epoch)
                loss_arr.append(loss)
                lr_arr.append(lr)
            elif '[EVAL] #Images: ' in line:
                g_dice = line.split('global Dice=')[1][1:].split(']')[0].split(' ')
                p_dice = line.split('per Dice=')[1][1:].split(']')[0].split(' ')

                eval_arr_g.append(g_dice)
                eval_arr_p.append(p_dice)

        eval_arr_g = np.array(eval_arr_g).astype(np.float)
        eval_arr_p = np.array(eval_arr_p).astype(np.float)

        fontsize = 4
        plt.tight_layout()
        plt.rcParams["figure.figsize"] = (30, 20)
        color_arr = ['r', 'g', 'b', 'y', 'c', 'm']
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        ax1 = plt.subplot(221)
        ax1.plot(np.array(epoch_arr), np.array(loss_arr), color_arr[3] + '-.')
        ax1.set_xlabel('epoch', fontsize=fontsize)
        ax1.set_ylabel('loss', fontsize=fontsize)
        ax1.xaxis.set_tick_params(labelsize=fontsize)
        ax1.yaxis.set_tick_params(labelsize=fontsize)
        ax1.set_title('Curve of Training Loss', fontsize=fontsize)

        ax2 = plt.subplot(222)
        ax2.plot(np.array(epoch_arr), np.array(lr_arr), color_arr[4] + '-.')
        ax2.set_xlabel('epoch', fontsize=fontsize)
        ax2.set_ylabel('lr', fontsize=fontsize)
        ax2.xaxis.set_tick_params(labelsize=fontsize)
        ax2.yaxis.set_tick_params(labelsize=fontsize)
        ax2.set_title('Curve of Learning Rate', fontsize=fontsize)

        ax3 = plt.subplot(223)
        for i in range(eval_arr_g.shape[1]):
            ax3.plot(np.array(range(len(eval_arr_g))) * 200, np.array(eval_arr_g)[:, i], color_arr[i] + '-.')

        ax3.set_xlabel('epoch', fontsize=fontsize)
        ax3.set_ylabel('Dice', fontsize=fontsize)
        ax3.xaxis.set_tick_params(labelsize=fontsize)
        ax3.yaxis.set_tick_params(labelsize=fontsize)
        ax3.set_title('Curve of global Dice', fontsize=fontsize)

        leg = plt.legend(labels=('background', 'liver', 'tumor'))
        # get the lines and texts inside legend box
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        # bulk-set the properties of all lines and texts
        plt.setp(leg_lines, linewidth=1)
        plt.setp(leg_texts, fontsize=2)

        ax4 = plt.subplot(224)
        for i in range(eval_arr_p.shape[1]):
            ax4.plot(np.array(range(len(eval_arr_p))) * 200, np.array(eval_arr_p)[:, i], color_arr[i] + '-.')

        ax4.set_xlabel('epoch', fontsize=fontsize)
        ax4.set_ylabel('Dice', fontsize=fontsize)
        ax4.xaxis.set_tick_params(labelsize=fontsize)
        ax4.yaxis.set_tick_params(labelsize=fontsize)
        ax4.set_title('Curve of per Dice', fontsize=fontsize)

        leg = plt.legend(labels=('background', 'liver', 'tumor'))
        # get the lines and texts inside legend box
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        # bulk-set the properties of all lines and texts
        plt.setp(leg_lines, linewidth=1)
        plt.setp(leg_texts, fontsize=2)

        # # 趋势线
        # for i in range(eval_arr_g.shape[1]):
        #     z = np.polyfit(np.array(range(len(eval_arr_g))) * 200, np.array(eval_arr_g)[:, i], 1)
        #     p = np.poly1d(z)
        #     ax3.plot(np.array(range(len(eval_arr_g))) * 200, p(np.array(range(len(eval_arr_g))) * 200), "m-")

        # # 趋势线
        # for i in range(eval_arr_p.shape[1]):
        #     z = np.polyfit(np.array(range(len(eval_arr_p))) * 200, np.array(eval_arr_p)[:, i], 1)
        #     p = np.poly1d(z)
        #     ax4.plot(np.array(range(len(eval_arr_p))) * 200, p(np.array(range(len(eval_arr_p))) * 200), "m-")

        plt.savefig(os.path.join(self.log_dir, 'show.png'), dpi=350)


if __name__ == "__main__":
    log_dir = sys.argv[1]

    log_visual = LogVisual(log_dir)
    log_visual.read_vis()