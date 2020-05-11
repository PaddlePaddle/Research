from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import shutil

import os

import argparse
import functools
import numpy as np
from reid.data.source import Dataset
from reid.data.reader_mt import create_readerMT
from config import cfg, parse_args, print_arguments, print_arguments_dict 
from reid.model import model_creator
import paddle.fluid as fluid
from reid.learning_rate import exponential_with_warmup_decay
from reid.cos_anneal_learning_rate import cos_anneal_with_warmup_decay
from reid.loss import triplet_loss
import time

def optimizer_build(cfg):
    momentum_rate = cfg.momentum
    weight_decay = cfg.weight_decay
    learning_rate = cfg.learning_rate
    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]
    print(cfg.lr_steps)
    lr = cos_anneal_with_warmup_decay(learning_rate, boundaries, values, warmup_iter = cfg.warm_up_iter, warmup_factor = 0.0001)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)
    return optimizer, lr


def calc_loss(logit, label, class_dim=1695, use_label_smoothing=True, epsilon=0.1):                                                                                    
    softmax_out = fluid.layers.softmax(logit)
    if use_label_smoothing:
        label_one_hot = fluid.layers.one_hot(input=label, depth=class_dim)
        smooth_label = fluid.layers.label_smooth(label=label_one_hot, epsilon=epsilon, dtype="float32")
        loss = fluid.layers.cross_entropy(input=softmax_out, label=smooth_label, soft_label=True)
    else:
        loss = fluid.layers.cross_entropy(input=softmax_out, label=label)
        
    return loss, softmax_out

def build_train_program(main_prog, startup_prog, cfg):
    cfg.use_multi_branch = True
    model = model_creator(cfg)
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            image = fluid.data(name='image', dtype='float32', shape=[None, 3, cfg.target_height, cfg.target_width])
            label = fluid.data(name='label', dtype='int64',   shape=[None, 1])
            data_loader = fluid.io.DataLoader.from_generator(feed_list=[image, label], capacity=512, use_double_buffer=True, iterable=False)
            
            x3_g_pool_fc, x4_g_pool_fc, x4_p_pool_fc, x3_g_avg, x3_g_max, x4_g_avg, x4_g_max, x4_p_avg, x4_p_max = model.net_multi_branch(input=image, class_dim=cfg.train_class_num, is_train=True, num_features = cfg.num_features)
        
            cost_1, pred_1 = fluid.layers.softmax_with_cross_entropy(x3_g_pool_fc, label, return_softmax=True)
            avg_cost_1 = fluid.layers.mean(x=cost_1)

            cost_2, pred_2 = fluid.layers.softmax_with_cross_entropy(x4_g_pool_fc, label, return_softmax=True)
            avg_cost_2 = fluid.layers.mean(x=cost_2)

            cost_3, pred_3 = fluid.layers.softmax_with_cross_entropy(x4_p_pool_fc, label, return_softmax=True)
            avg_cost_3 = fluid.layers.mean(x=cost_3)
            
            cost_4 = triplet_loss.tripletLoss(x3_g_avg, label, args.batch_size)
            avg_cost_4 = fluid.layers.mean(x=cost_4)

            cost_5 = triplet_loss.tripletLoss(x3_g_max, label, args.batch_size)
            avg_cost_5 = fluid.layers.mean(x=cost_5)
            
            cost_6 = triplet_loss.tripletLoss(x4_g_avg, label, args.batch_size)
            avg_cost_6 = fluid.layers.mean(x=cost_6)

            cost_7 = triplet_loss.tripletLoss(x4_g_max, label, args.batch_size)
            avg_cost_7 = fluid.layers.mean(x=cost_7)
 
            cost_8 = triplet_loss.tripletLoss(x4_p_avg, label, args.batch_size)
            avg_cost_8 = fluid.layers.mean(x=cost_8)
    
            cost_9 = triplet_loss.tripletLoss(x4_p_max, label, args.batch_size)
            avg_cost_9 = fluid.layers.mean(x=cost_9)

            total_cost = (avg_cost_1 + avg_cost_2 + avg_cost_3) / 3.0 + (avg_cost_4 + avg_cost_5 + avg_cost_6 + avg_cost_7 + avg_cost_8 + avg_cost_9) / 6.0  
            
            acc_1 = fluid.layers.accuracy(input=pred_1, label=label, k=1)
            acc_2 = fluid.layers.accuracy(input=pred_2, label=label, k=1)
            acc_3 = fluid.layers.accuracy(input=pred_3, label=label, k=1)

            build_program_out = [data_loader, total_cost, acc_1, acc_2, acc_3]

            optimizer, learning_rate = optimizer_build(cfg)
            optimizer.minimize(total_cost)

            build_program_out.append(learning_rate)

    return build_program_out




def main(cfg):
    ReidDataset = Dataset(root = cfg.data_dir)
    if cfg.use_crop:
        ReidDataset.load_trainval('all_trainval_pids_crop.txt')
    else:
        ReidDataset.load_trainval('all_trainval_pids.txt')

    reader_config = {'dataset':ReidDataset.train, 
                     'img_dir':'./dataset/aicity20_all/',
                     'batch_size':cfg.batch_size,
                     'num_instances':cfg.num_instances,
                     'sample_type':'Identity',
                     'shuffle':True,
                     'drop_last':True,
                     'worker_num':8,
                     'use_process':True,
                     'bufsize':32,
                     'cfg':cfg,
                     'input_fields':['image','pid']}

    devices_num = fluid.core.get_cuda_device_count()
    print("Found {} CUDA devices.".format(devices_num))
    
    new_reader, num_classes, num_batch_pids, num_iters_per_epoch = create_readerMT(reader_config, max_iter=cfg.max_iter*devices_num)
    #pdb.set_trace()


    assert cfg.batch_size % cfg.num_instances == 0

    num_iters_per_epoch = int(num_iters_per_epoch / devices_num)
    print('per epoch contain iterations:', num_iters_per_epoch)
    max_epoch = int(cfg.max_iter / num_iters_per_epoch)

    cfg.train_class_num = num_classes

    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    train_reader, total_cost, acc_1, acc_2, acc_3, lr_node = build_train_program(main_prog=train_prog, startup_prog=startup_prog, cfg=cfg)
    total_cost.persistable = True
    acc_1.persistable = True
    acc_2.persistable = True
    acc_3.persistable = True
    train_fetch_vars = [total_cost, lr_node, acc_1, acc_2, acc_3]

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    def save_model(exe, postfix, prog):
        model_path = os.path.join(cfg.model_save_dir, cfg.model_arch, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        else:
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path, main_program=prog)
    if cfg.pretrain:
        print(cfg.pretrain)
        def if_exist(var):
            if os.path.exists(os.path.join(cfg.pretrain, var.name)):
                print(var.name)
                return True
            else:
                return False
        fluid.io.load_vars(
            exe, cfg.pretrain, main_program=train_prog, predicate=if_exist)

    compile_program = fluid.compiler.CompiledProgram(train_prog).with_data_parallel(loss_name=total_cost.name)
    if devices_num==1:
        places = fluid.cuda_places(0)
    else:
        places = fluid.cuda_places()
    train_reader.set_sample_list_generator(new_reader, places=places)
    train_reader.start()

    try:
        start_time = time.time()
        snapshot_loss = 0
        snapshot_time = 0

        for cur_iter in range(cfg.start_iter, cfg.max_iter):
            cur_peoch = int(cur_iter / num_iters_per_epoch)
            outputs = exe.run(compile_program, fetch_list=[v.name for v in train_fetch_vars])
            cur_loss = np.mean(np.array(outputs[0]))
            cur_lr = np.mean(np.array(outputs[1]))
            cur_acc_1 = np.mean(np.array(outputs[2]))
            cur_acc_2 = np.mean(np.array(outputs[3]))
            cur_acc_3 = np.mean(np.array(outputs[4]))

            snapshot_loss += cur_loss

            cur_time = time.time() - start_time
            start_time = time.time()
            snapshot_time += cur_time

            output_str = 'epoch {}/{}, iter {}/{}, lr:{:.6f}, total loss:{:.4f}, accuracy_1:{:.4f}, accuracy_2:{:.4f}, accuracy_3:{:.4f}, time:{} '.format(cur_peoch, max_epoch, cur_iter, cfg.max_iter, cur_lr, cur_loss, cur_acc_1, cur_acc_2, cur_acc_3, cur_time)
            print(output_str)

            if (cur_iter + 1) % cfg.snapshot_iter == 0:
                save_model(exe,"model_iter{}".format(cur_iter),train_prog)
                print("Snapshot {} saved, average loss: {}, \
                      average time: {}".format(
                    cur_iter + 1, snapshot_loss / float(cfg.snapshot_iter),
                    snapshot_time / float(cfg.snapshot_iter)))

                snapshot_loss = 0
                snapshot_time = 0

    except fluid.core.EOFException:
        train_reader.reset()

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    print_arguments_dict(args)
    main(args)
