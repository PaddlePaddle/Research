# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = "0.99"
import sys
import timeit
import time
import argparse
import pprint
import shutil
import functools
import math
import logging
import numpy as np
import matplotlib
import random
from datetime import datetime

import paddle
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.fluid.dygraph.base import to_variable
from src.utils.config import cfg
from src.utils.timer import Timer, calculate_eta
from src.datasets.cityscapes import cityscapes_train, cityscapes_quick_val 
from src.datasets.pascal_context import pascal_context_train, pascal_context_quick_val
from src.utils.solver import Lr
from src.models.modeling.pspnet import PSPNet
from src.models.modeling.glore import Glore
from src.utils.iou import IOUMetric
from src.utils.loss import multi_cross_entropy_loss
from PIL import ImageOps, Image, ImageEnhance, ImageFilter


def print_info(*msg):
    if cfg.TRAINER_ID == 0:
        print(*msg)


def parse_args():
    parser = argparse.ArgumentParser(description='semseg-paddle')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str)
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Use gpu or cpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_mpio',
        dest='use_mpio',
        help='Use multiprocess I/O or not',
        action='store_true',
        default=False)
    parser.add_argument(
        '--log_steps',
        dest='log_steps',
        help='Display logging information at every log_steps',
        default=10,
        type=int)
    parser.add_argument(
        '--debug',
        dest='debug',
        help='debug mode, display detail information of training',
        action='store_true')
    parser.add_argument(
        '--use_tb',
        dest='use_tb',
        help='whether to record the data during training to Tensorboard',
        action='store_true')
    parser.add_argument(
        '--tb_log_dir',
        dest='tb_log_dir',
        help='Tensorboard logging directory',
        default=None,
        type=str)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Evaluation models result on every new checkpoint',
        action='store_true')
    parser.add_argument(
        'opts',
        help='See utils/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    return parser.parse_args()

def get_model(cfg):
    if cfg.MODEL.MODEL_NAME.lower() == 'pspnet':
        model = PSPNet(cfg.TRAIN.PRETRAINED_MODEL_FILE, cfg.DATASET.DATA_DIM, cfg.DATASET.NUM_CLASSES, multi_grid=cfg.MODEL.BACKBONE_MULTI_GRID)
    elif cfg.MODEL.MODEL_NAME.lower() == 'glore':
        model = Glore(cfg.TRAIN.PRETRAINED_MODEL_FILE, cfg.DATASET.DATA_DIM, cfg.DATASET.NUM_CLASSES, multi_grid=cfg.MODEL.BACKBONE_MULTI_GRID)
    return model


def get_data(cfg, split, base_size, crop_size, batch_size=1, gpu_num=1):
    if cfg.DATASET.DATASET_NAME.lower() in ['cityscapes', 'pascal_context']:
        dataset_name = cfg.DATASET.DATASET_NAME.lower()
        data_loader = globals()[dataset_name + '_' + split]
        return data_loader(root=cfg.DATASET.DATA_ROOT,
                            base_size=base_size,
                            crop_size=crop_size,
                            scale=True,
                            xmap=True,
                            batch_size=batch_size,
                            gpu_num=gpu_num)
    else:
        raise ValueError('Dataset is not supported, please check!')

def get_loss_weight(cfg):
    if cfg.DATASET.DATASET_NAME.lower() == 'cityscapes':
        weight = np.array([0.8373, 0.918, 0.866, 1.0345, 
                  1.0166, 0.9969, 0.9754, 1.0489,
                  0.8786, 1.0023, 0.9539, 0.9843, 
                  1.1116, 0.9037, 1.0865, 1.0955, 
                  1.0865, 1.1529, 1.0507]).astype('float32')
    elif cfg.DATASET.DATASET_NAME.lower() == 'pascal_context':
        weight = np.array([0.9752, 1.1304, 1.0394, 0.9790, 1.1678, 0.9746, 0.9684, 0.9843, 1.0779,
                  1.0104, 0.8645, 0.9524, 0.9868, 0.9177, 0.8935, 0.9964, 0.9434, 0.9809,
                  1.1404, 0.9986, 1.1305, 1.0130, 0.9012, 1.0040, 0.9556, 0.9000, 1.0835,
                  1.1341, 0.8632, 0.8645, 0.9675, 1.1404, 1.1137, 0.9552, 0.9701, 1.4059,
                  0.8564, 1.1347, 1.0534, 0.9957, 0.9114, 1.0241, 0.9884, 1.0245, 1.0236,
                  1.1080, 0.8488, 1.0122, 0.9343, 0.9747, 1.0404, 0.9482, 0.8689, 1.1267,
                  0.9776, 0.8640, 0.9030, 0.9728, 1.0239]).astype('float32')
    return fluid.dygraph.to_variable(weight)


def optimizer_setting(model):
    if cfg.SOLVER.WEIGHT_DECAY is not None:
        regular = fluid.regularizer.L2Decay(regularization_coeff=cfg.SOLVER.WEIGHT_DECAY)
    else:
        regular = None
    if cfg.SOLVER.LR_POLICY == 'poly':
        step_per_epoch = int(math.ceil(cfg.DATASET.TRAIN_TOTAL_IMAGES / cfg.TRAIN_BATCH_SIZE))
        lr_scheduler = Lr(lr_policy='poly',
                          base_lr=cfg.SOLVER.LR,
                          epoch_nums=cfg.SOLVER.NUM_EPOCHS ,
                          step_per_epoch=step_per_epoch,
                          power=cfg.SOLVER.POWER,
                          warm_up=cfg.SOLVER.LR_WARMUP,
                          warmup_epoch=cfg.SOLVER.LR_WARMUP_STEPS)
        decayed_lr = lr_scheduler.get_lr()
    elif cfg.SOLVER.LR_POLICY == 'cosine':
        lr_scheduler = Lr(lr_policy='cosine',
                          base_lr=0.003,
                          epoch_nums=cfg.SOLVER.NUM_EPOCHS ,
                          step_per_epoch=cfg.DATASET.TRAIN_TOTAL_IMAGES//cfg.TRAIN_BATCH_SIZE,
                          warm_up=cfg.SOLVER.LR_WARMUP,
                          warmup_epoch=cfg.SOLVER.LR_WARMUP_STEPS)
        decayed_lr = lr_scheduler.get_lr()
    elif cfg.SOLVER.LR_POLICY == 'piecewise':
        lr_scheduler = Lr(lr_policy='piecewise',
                          base_lr=0.003,
                          epoch_nums=cfg.SOLVER.NUM_EPOCHS ,
                          step_per_epoch=cfg.DATASET.TRAIN_TOTAL_IMAGES//cfg.TRAIN_BATCH_SIZE,
                          warm_up=cfg.SOLVER.LR_WARMUP,
                          warmup_epoch=cfg.SOLVER.LR_WARMUP_STEPS,
                          decay_epoch=[50, 100, 150],
                          gamma=0.1)
        decayed_lr = lr_scheduler.get_lr()
    else:
        decayed_lr = 0.003

    return fluid.optimizer.MomentumOptimizer(learning_rate=decayed_lr,
                                             momentum=cfg.SOLVER.MOMENTUM,
                                             parameter_list=model.parameters(),
                                             regularization=regular)


def mean_iou(pred, label, iou):
    n, c, h, w = pred.shape
    pred = fluid.layers.cast(fluid.layers.argmax(pred, axis=1), 'int32')
    pred_np = pred.numpy()
    label_np = label.numpy()
    for i in range(n):
        pred = pred_np[i, :, :]
        label = label_np[i, :, :]
        iou.add_batch(pred, label)
    return iou



def train(cfg):
    np.random.seed(1)
    random.seed(1)
    drop_last = True
    print_info("#train_batch_size: {}".format(cfg.TRAIN_BATCH_SIZE))
    print_info("#batch_size_per_dev: {}".format(cfg.TRAIN_BATCH_SIZE_PER_GPU))
    gpu_num = int(float(cfg.TRAIN_BATCH_SIZE)/float(cfg.TRAIN_BATCH_SIZE_PER_GPU))
    print_info("Used GPU number:", gpu_num)

    place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id) if args.use_gpu else fluid.CPUPlace()
    
    train_loss_title = 'Train_loss'
    test_loss_title = 'Test_loss'

    train_iou_title = 'Train_mIOU'
    test_iou_title = 'Test_mIOU'


    with fluid.dygraph.guard(place):

        strategy = fluid.dygraph.prepare_context()

        train_data = get_data(cfg=cfg,
                              split='train',
                              base_size=cfg.DATAAUG.BASE_SIZE,
                              crop_size=cfg.DATAAUG.CROP_SIZE,
                              batch_size=cfg.TRAIN_BATCH_SIZE_PER_GPU,
                              gpu_num=gpu_num)
        
        val_data = get_data(cfg=cfg,
                            split='quick_val',
                            base_size=cfg.DATAAUG.BASE_SIZE,
                            crop_size=cfg.DATAAUG.CROP_SIZE,
                            batch_size=cfg.VAL_BATCH_SIZE,
                            gpu_num=gpu_num)

        batch_train_data = paddle.batch(paddle.reader.shuffle(
            train_data, buf_size=cfg.TRAIN_BATCH_SIZE_PER_GPU * 64),
            batch_size=cfg.TRAIN_BATCH_SIZE_PER_GPU,
            drop_last=True)     

        batch_val_data = paddle.batch(val_data, batch_size=cfg.VAL_BATCH_SIZE, drop_last=False)  

        batch_train_data = fluid.contrib.reader.distributed_batch_reader(batch_train_data)

        train_avg_loss_manager = fluid.metrics.Accuracy()
        test_avg_loss_manager = fluid.metrics.Accuracy()

        better_miou_train = 0
        better_miou_test = 0

        model = get_model(cfg)

        optimizer = optimizer_setting(model)

        model = fluid.dygraph.DataParallel(model, strategy)
        
        loss_weight = get_loss_weight(cfg)
        
        for epoch in range(cfg.SOLVER.NUM_EPOCHS):
            prev_time = datetime.now()
            train_avg_loss_manager.reset()
            train_iou_np = IOUMetric(cfg.DATASET.NUM_CLASSES)
            val_iou_np = IOUMetric(cfg.DATASET.NUM_CLASSES)
            model.train()
            for batch_id, data in enumerate(batch_train_data()):

                start_time = time.time()
                
                image = np.array([x[0] for x in data]).astype('float32')
                
                label = np.array([x[1] for x in data]).astype('int64')
            
                image = fluid.dygraph.to_variable(image)
                label = fluid.dygraph.to_variable(label)
                label.stop_gradient = True
            
                logits=model(image)
                
                train_loss = multi_cross_entropy_loss(logits[0], logits[1], label, cfg.DATASET.NUM_CLASSES, loss_weight)
                train_avg_loss = fluid.layers.mean(train_loss)

                train_iou_np = mean_iou(logits[0], label, train_iou_np)
                train_avg_loss = model.scale_loss(train_avg_loss)
                train_avg_loss.backward()
                model.apply_collective_grads()
                
                optimizer.minimize(train_avg_loss)
                lr = optimizer.current_step_lr()
                model.clear_gradients()

                train_avg_loss_manager.update(train_avg_loss.numpy(), weight=int(cfg.TRAIN_BATCH_SIZE))
                end_time = time.time()
                during_time = str(end_time - start_time)
                batch_train_str = "trainer_id:{}, epoch: {}, time:{}s, batch: {}, train_avg_loss: {:.6f}, " \
                                  "current_lr: {}.".format(int(os.getenv("PADDLE_TRAINER_ID", 0)),
                                                               epoch + 1,
                                                               during_time[:6],
                                                               batch_id + 1,
                                                               float(train_avg_loss),
                                                               lr)
                
                if batch_id % 100 == 0:
                    print(batch_train_str)
                finish_time = time.time()

            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            acc, acc_cls, iu, mean_iu_train, fwavacc, kappa = train_iou_np.evaluate()
            time_str = " Time %02d:%02d:%02d" % (h, m, s)
            train_str = "\nepoch: {}, train_avg_loss: {:.6f}, " \
                        "new_miou:{}".format(epoch + 1,
                                            train_avg_loss_manager.eval()[0],
                                            mean_iu_train)
            print(train_str + time_str + '\n')

            train_id = int(os.getenv("PADDLE_TRAINER_ID", 0))

            test_avg_loss_manager.reset()
            model.eval()
            for batch_id, data in enumerate(batch_val_data()):
                start_time = time.time()
                image = np.array([x[0] for x in data]).astype('float32')
                label = np.array([x[1] for x in data]).astype('int64')
                image = fluid.dygraph.to_variable(image)
                label = fluid.dygraph.to_variable(label)
        
                label.stop_gradient = True
                
                logits = model(image)
                
                val_loss = multi_cross_entropy_loss(logits[0], logits[1], label, cfg.DATASET.NUM_CLASSES, loss_weight)
                val_avg_loss = fluid.layers.mean(val_loss)

                val_iou_np = mean_iou(logits[0], label, val_iou_np)
                test_avg_loss_manager.update(val_avg_loss.numpy(), weight=int(cfg.VAL_BATCH_SIZE))
                end_time = time.time()
                during_time = str(end_time - start_time)
                if batch_id % 100 == 0:
                    print('Validation trained_id:{}, epoch:{}, batch_id:{}, time:{}s, val_loss:{}.'.format(train_id, epoch+1,
                                                                                                            batch_id,
                                                                                                            during_time[:6],
                                                                                                            float(val_avg_loss),
                                                                                                            ))
            acc, acc_cls, iu, mean_iu_val, fwavacc, kappa = val_iou_np.evaluate()
            print('Validation epoch:{}, val_loss{}, val_iou_np:{}'.format(epoch+1, test_avg_loss_manager.eval()[0], mean_iu_val))
            if train_id == 1:
                fluid.dygraph.save_dygraph(model.state_dict(), cfg.TRAIN.MODEL_SAVE_DIR + 'model_' + str(epoch))
                fluid.dygraph.save_dygraph(model.state_dict(), cfg.TRAIN.MODEL_SAVE_LAST)
                print('successfully saved the last model!')

            if better_miou_train < mean_iu_val:
                better_miou_train = mean_iu_val
                fluid.dygraph.save_dygraph(model.state_dict(), cfg.TRAIN.MODEL_SAVE_BEST + '_' + str(train_id))
                print('successfully saved the best model!')
        print('best iou:', better_miou_train) # best_mean_iou for quick validation, not the real iou 
        print('Done -------------------------------')


def main(args):
    if args.cfg_file is not None:
        cfg.update_from_file(args.cfg_file)
    if args.opts:
        cfg.update_from_list(args.opts)

    cfg.TRAINER_ID = int(os.getenv("PADDLE_TRAINER_ID", 0))
    cfg.NUM_TRAINERS = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    cfg.check_and_infer()
    print_info(pprint.pformat(cfg))
    train(cfg)
    

if __name__ == '__main__':
    args = parse_args()
    start = timeit.default_timer()
    main(args)
    end = timeit.default_timer()
    print("training time: {} h".format(1.0*(end-start)/3600))
    
