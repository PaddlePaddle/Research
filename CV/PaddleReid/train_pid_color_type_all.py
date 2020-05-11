from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import shutil
import os
import time
import argparse
import functools
import numpy as np

import paddle.fluid as fluid

from reid.cos_anneal_learning_rate import cos_anneal_with_warmup_decay
from reid.data.reader_mt import create_readerMT
from reid.data.source.dataset import Dataset
from reid.model import model_creator
from reid.learning_rate import exponential_with_warmup_decay
from reid.loss.triplet_loss import tripletLoss

from config import cfg, parse_args, print_arguments, print_arguments_dict 

def optimizer_build(cfg):
    momentum_rate = cfg.momentum
    weight_decay = cfg.weight_decay
    learning_rate = cfg.learning_rate
    boundaries = cfg.lr_steps
    #pdb.set_trace()
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]
    print(cfg.lr_steps)
    start_lr = learning_rate
    lr = cos_anneal_with_warmup_decay(start_lr, boundaries, values, warmup_iter = cfg.warm_up_iter, warmup_factor = 0.0001)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)
    return optimizer, lr


def build_train_program(main_prog, startup_prog, cfg):
    model = model_creator(cfg)
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            image = fluid.data(name='image', dtype='float32', shape=[None, 3, cfg.target_height, cfg.target_width])
            pid = fluid.data(name='pid', dtype='int64',   shape=[None, 1])
            colorid = fluid.data(name='colorid', dtype='int64',   shape=[None, 1])
            typeid = fluid.data(name='typeid', dtype='int64',   shape=[None, 1])
            data_loader = fluid.io.DataLoader.from_generator(feed_list=[image, pid, colorid, typeid], capacity=128, use_double_buffer=True, iterable=False)

            pid_out, color_out, type_out, reid_feature = model.net_pid_color_type(input=image, class_dim=cfg.train_class_num, is_train=True, num_features = cfg.num_features)

            pid_softmax_out = fluid.layers.softmax(pid_out, use_cudnn=False)
            pid_cost = fluid.layers.cross_entropy(input=pid_softmax_out, label=pid, ignore_index=-1)
            pid_cost = fluid.layers.reduce_mean(pid_cost)
            
            #triplet_cost = tripletLoss(reid_feature, pid, batch_size = cfg.batch_size, margin=cfg.margin, num_instances = cfg.num_instances)
            #triplet_cost = fluid.layers.reduce_mean(triplet_cost)

            color_softmax_out = fluid.layers.softmax(color_out, use_cudnn=False)
            color_cost = fluid.layers.cross_entropy(input=color_softmax_out, label=colorid, ignore_index=-1)
            color_cost = fluid.layers.reduce_mean(color_cost)

            type_softmax_out = fluid.layers.softmax(type_out, use_cudnn=False)
            type_cost = fluid.layers.cross_entropy(input=type_softmax_out, label=typeid, ignore_index=-1)
            type_cost = fluid.layers.reduce_mean(type_cost)

            #avg_cost = pid_cost + triplet_cost + 0.1*color_cost + 0.1*type_cost
            avg_cost = pid_cost + 0.1*color_cost + 0.1*type_cost
            build_program_out = [data_loader, avg_cost, pid_cost, color_cost, type_cost]

            optimizer, learning_rate = optimizer_build(cfg)
            optimizer.minimize(avg_cost)
            build_program_out.append(learning_rate)

    return build_program_out




def main(cfg):
    #pdb.set_trace()
    count = 0

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
                     'input_fields':['image','pid','colorid', 'typeid']}

    devices_num = fluid.core.get_cuda_device_count()
    print("Found {} CUDA devices.".format(devices_num))

    new_reader, num_classes, num_batch_pids, num_iters_per_epoch = create_readerMT(reader_config, max_iter=cfg.max_iter*devices_num)

    assert cfg.batch_size % cfg.num_instances == 0

    num_iters_per_epoch = int(num_iters_per_epoch / devices_num)
    print('per epoch contain iterations:', num_iters_per_epoch)
    max_epoch = int(cfg.max_iter / num_iters_per_epoch)


    cfg.train_class_num = num_classes
    print("num_pid: ", cfg.train_class_num)


    startup_prog = fluid.Program()
    train_prog = fluid.Program()


    train_reader, avg_cost, pid_cost, color_cost, type_cost, lr_node = build_train_program(main_prog=train_prog, startup_prog=startup_prog, cfg=cfg)
    avg_cost.persistable = True
    pid_cost.persistable = True
    color_cost.persistable = True
    type_cost.persistable = True
    train_fetch_vars = [avg_cost, pid_cost, color_cost, type_cost, lr_node]



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

    compile_program = fluid.compiler.CompiledProgram(train_prog).with_data_parallel(loss_name=avg_cost.name)
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
            losses = exe.run(compile_program, fetch_list=[v.name for v in train_fetch_vars])
            cur_loss = np.mean(np.array(losses[0]))
            cur_pid_loss = np.mean(np.array(losses[1]))
            cur_color_loss = np.mean(np.array(losses[2]))
            cur_type_loss = np.mean(np.array(losses[3]))
            cur_lr = np.mean(np.array(losses[4]))
            # cur_lr = np.array(fluid.global_scope().find_var('learning_rate').get_tensor())

            snapshot_loss += cur_loss

            cur_time = time.time() - start_time
            start_time = time.time()
            snapshot_time += cur_time
            #pdb.set_trace()


            output_str = '{}/{}epoch, {}/{}iter, lr:{:.6f}, loss:{:.4f}, pid:{:.4f}, color:{:.4f}, type:{:.4f}, time:{} '.format(cur_peoch, max_epoch, cur_iter, cfg.max_iter, cur_lr, cur_loss, cur_pid_loss, cur_color_loss, cur_type_loss, cur_time )
            print(output_str)
            #fluid.io.save_inference_model(cfg.model_save_dir+'/infer_model', infer_node.name, pred_list, exe, main_program=train_prog, model_filename='model', params_filename='params')

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

    save_model(exe, 'model_final', train_prog)
    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    print_arguments_dict(args)
    main(args)
