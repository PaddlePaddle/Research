######################################################################
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
######################################################################
"""
File: 
"""

import paddle
import os
import paddle.fluid as fluid
import numpy as np
import sys
import math
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import paddle.fluid.dygraph.base as base
random.seed(42)


class Config(object):
    """Config"""
    def __init__(self):
        self.CLASS_SIZE = 2
        self.EMBED_SIZE = 128
        self.HIDDEN_SIZE = 128
        self.STACKED_NUM = 5
        self.INPUT_SIZE = 1363
        self.TRAIN_RATE = 0.7
        self.VAL_RATE = 0.15
        self.SHUFFLE = False
        self.GOAL_FILE = False
        self.DOWN_SAMPLING = False
        self.BATCH_SIZE = 256
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCH = 100
        self.DEVICE = fluid.CPUPlace()
        self.SAVE_PATH = "../model_state/paddle_astar_kg_v1.mdl"


def file_reader(file_path):
    """file_reader"""
    data = None
    with open(file_path, "r") as f:
        data = eval(f.read())
        f.close()
    return data


def reader_generater(datas):
    """reader_generater"""
    def reader():
        """reader"""
        for data in datas:
            yield data
    return reader


def get_data(file_path, data_tag, test=False):
    x = file_reader(file_path + data_tag + "_next_goal_entity.txt")
    if test is False:
        y = file_reader(file_path + data_tag + "_next_goal_entity_label.txt")
    g = file_reader(file_path + data_tag + "_final_goal_entity.txt")

    idx = file_reader(file_path + data_tag + "_next_goal_entity_idx.txt")
    c = [item[-1] for item in x]
    
    data = None
    if test is False:
        data = [[x[i], y[i], c[i], g[i], idx[i], 1] for i in range(len(x))]
    else:
        data = [[x[i], c[i], g[i][0], idx[i], 1] for i in range(len(x))]
    return data


def get_point_data(train_rate, val_rate, batch_size, train_down_sampling=False, goal_file=True):
    """get_point_data"""
    
    file_path = "../train_data/"
    train_data = get_data(file_path, "train")
    val_data = get_data(file_path, "val")
    test_data = get_data(file_path, "test", test=True)
    
    train_len = len(train_data)
    val_len = len(val_data)
    data_len = train_len + val_len + len(test_data)
    
    print(len(train_data), len(val_data), len(test_data))
    train_reader = paddle.batch(reader_generater(train_data), batch_size=batch_size)
    val_reader = paddle.batch(reader_generater(val_data), batch_size=batch_size)
    test_reader = paddle.batch(reader_generater(test_data), batch_size=batch_size)
    
    return train_reader, val_reader, test_reader


def a_star(sequence, current, goal, config):
    """a_star"""
    input_dim = config.INPUT_SIZE
    class_dim = config.CLASS_SIZE
    embed_dim = config.EMBED_SIZE
    hidden_dim = config.HIDDEN_SIZE
    stacked_num = config.STACKED_NUM
    
    weight_data = np.random.random(size=(input_dim, embed_dim))
    my_param_attrs = fluid.ParamAttr(
        name="embedding",
        learning_rate=config.LEARNING_RATE,
        initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
        trainable=True
    )
    
    seq_embed = fluid.embedding(
        input=sequence, size=[input_dim, embed_dim], param_attr=my_param_attrs)
    curr_embed = fluid.embedding(
        input=current, size=[input_dim, embed_dim], param_attr=my_param_attrs)
    goal_embed = fluid.embedding(
        input=goal, size=[input_dim, embed_dim], param_attr=my_param_attrs)
    
    fc1 = fluid.layers.fc(input=seq_embed, size=hidden_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hidden_dim)
    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hidden_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hidden_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]
    
    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')
    
    current_cost_embed = [fc_last, lstm_last, curr_embed]
    remain_cost_embed = [fc_last, lstm_last, goal_embed]
    
    pred_curr_fc1 = fluid.layers.fc(input=current_cost_embed, size=64, act="relu")
    current_cost = fluid.layers.fc(input=pred_curr_fc1, size=1, act="sigmoid")
    
    pred_goal_fc1 = fluid.layers.fc(input=remain_cost_embed, size=64, act="relu")
    remain_cost = fluid.layers.fc(input=pred_goal_fc1, size=1, act="sigmoid")
    
    prediction = 0.5 * current_cost + 0.5 * remain_cost
    return prediction


def inference_program(config):
    """inference_program"""
    sequence = fluid.data(name="sequence", shape=[None, 1], dtype="int64", lod_level=1)
    current = fluid.data(name="current", shape=[None], dtype="int64")
    goal = fluid.data(name="goal", shape=[None], dtype="int64")
    net = a_star(sequence, current, goal, config)
    return net


def train_program(prediction):
    """train_program"""
    label = fluid.data(name="label", shape=[None, 1], dtype="float32")
    idx = fluid.data(name="idx", shape=[None], dtype="int64")
    weight = fluid.data(name="weight", shape=[None, 1], dtype="float32")
    cost = fluid.layers.log_loss(input=prediction, label=label)
    cost = paddle.fluid.layers.elementwise_mul(weight, cost)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def train_accuracy(prediction, data):
    """train_accuracy"""
    y_pred = list()
    y_true = list()
    for item in prediction:
        if item[0] > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    for item in data:
        y_true.append(item[1])
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def argmax_metrics(y_pred, y_true, y_type, y_idx):
    """argmax_metrics"""
    group_data = list()
    
    idx = 0
    while idx < len(y_idx):
        tmp_pred = list()
        tmp_true = list()
        tmp_type = list()
        tmp_idx = list()
        
        previous = y_idx[idx]
        while idx < len(y_idx) and y_idx[idx] == previous:
            tmp_pred.append(y_pred[idx])
            tmp_true.append(y_true[idx])
            tmp_type.append(y_type[idx])
            tmp_idx.append(y_idx[idx])
            idx += 1
            
        group_data.append([tmp_pred, tmp_true, tmp_type, tmp_idx])
    
    data_pred = list()
    data_true = list()
    for item in group_data:
        true_type = None
        pred_type = None
        max_pred = -1
        for a, b, c in zip(item[0], item[1], item[2]):
            if a > max_pred:
                max_pred = a
                pred_type = c
            if b == 1:
                true_type = c
                
        if true_type != None and pred_type != None:
            data_pred.append(pred_type)
            data_true.append(true_type)

    all_true_set = set()
    for dt in data_true:
        all_true_set.add(dt)
#     for a, b in zip(data_pred, data_true):
#         print(a, b)
    return evaluation(data_pred, data_true)

       
def evaluation(y_pred, y_true, flag="macro"):
    """evaluation"""
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, average=flag), \
    precision_score(y_true, y_pred, average=flag), f1_score(y_true, y_pred, average=flag)


def optimizer_func(learning_rate):
    """optimizer_func"""
    return fluid.optimizer.Adagrad(learning_rate=learning_rate)

  
def train(config):
    """train"""
    print("Reading data...")
    train_reader, val_reader, test_reader = get_point_data(
        config.TRAIN_RATE, config.VAL_RATE, config.BATCH_SIZE, config.DOWN_SAMPLING, config.GOAL_FILE)
    
    feed_order = ["sequence", "label", "current", "goal", "idx", "weight"]
    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()
    main_program.random_seed = 90
    star_program.random_seed = 90
    
    prediction = inference_program(config)
    avg_cost = train_program(prediction)
    
    val_program = main_program.clone(for_test=True)
    
    sgd_optimizer = optimizer_func(config.LEARNING_RATE)
    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(config.DEVICE)
    
    def val_loop(program, reader):
        """val_loop"""
        feed_var_list = [program.global_block().var(var_name) for var_name in feed_order]
        feeder_var = fluid.DataFeeder(feed_list=feed_var_list, place=config.DEVICE)
        val_exe = fluid.Executor(config.DEVICE)
        val_pred, val_prob, val_true, val_type, val_idx = list(), list(), list(), list(), list()
        for val_data in reader():
            preds = val_exe.run(
                program=program,
                feed=feeder_var.feed(val_data),
                fetch_list=prediction
            )
            preds = np.squeeze(np.array(preds))
            for pred in preds:
                if pred <= 0.5:
                    val_pred.append(0)
                else:
                    val_pred.append(1)
                val_prob.append(pred)
            for vd in val_data:
                val_type.append(vd[0][-1])
                val_true.append(vd[1])
                val_idx.append(vd[-2])
        bi_acc, bi_rec, bi_pre, bi_f1 = evaluation(val_pred, val_true)
        ag_acc, ag_rec, ag_pre, ag_f1 = argmax_metrics(val_prob, val_true, val_type, val_idx)
        return bi_acc, bi_rec, bi_pre, bi_f1, ag_acc, ag_rec, ag_pre, ag_f1
    
    def train_loop():
        """train_loop"""
        feed_var_list_loop = [main_program.global_block().var(var_name) for var_name in feed_order]
        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=config.DEVICE)
        exe.run(star_program)
        
        max_epoch, max_acc = 0, 0
        for epoch_id in range(config.NUM_EPOCH):
            train_loss = list()
            train_acc = list()
            
            for batch_id, data in enumerate(train_reader()):
                metrics = exe.run(
                    main_program,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost, prediction]
                )
                train_loss.append(np.array(metrics[0]))
                train_acc.append(train_accuracy(np.array(metrics[1]), data))
            
            bi_acc, bi_rec, bi_pre, bi_f1, ag_acc, ag_rec, ag_pre, ag_f1 =  val_loop(val_program, val_reader)
            print("Epoch-%d, Train-Loss:%.4f, Train-Acc:%.4f, BI-Acc:%.4f, BI-Rec:%.4f, BI-Pre:%.4f, \
                BI-F1:%.4f, AG-Acc:%.4f, AG-Rec:%.4f, AG-Pre:%.4f, AG-F1:%.4f" % (
                epoch_id, np.mean(train_loss), np.mean(train_acc), 
                bi_acc, bi_rec, bi_pre, bi_f1, 
                ag_acc, ag_rec, ag_pre, ag_f1))
            
            if bi_acc > max_acc:
                max_acc = bi_acc
                max_epoch = epoch_id
                fluid.io.save_inference_model(config.SAVE_PATH, ["sequence", "current", "goal"], prediction, exe)
        print("max_epoch: %d, max_acc: %.4f" % (max_epoch, max_acc))
        
        
    train_loop()
    inference(config, test_reader)

  
def inference(config, reader):
    """inference"""
    exe = fluid.Executor(config.DEVICE)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inferencer, feed_target_names, fetch_targets] = fluid.io.load_inference_model(config.SAVE_PATH, exe)
        
        test_pred, test_prob, test_type, test_idx = list(), list(), list(), list()
        for data in reader():
            sequence = list()
            current = list()
            goal = list()
            for d in data:
                sequence.append(d[0])
                current.append(d[1])
                goal.append(d[2])
                
                test_type.append(d[0][-1])
                test_idx.append(d[-2])
                
            base_shape = [[len(seq) for seq in sequence]]
            lod_seq = fluid.create_lod_tensor(sequence, base_shape, config.DEVICE)

            new_data = {
                feed_target_names[0]: lod_seq,
                feed_target_names[1]: np.array(current),
                feed_target_names[2]: np.array(goal)
            }
            
            preds = exe.run(
                inferencer,
                feed=new_data,
                fetch_list=fetch_targets,
                return_numpy=False
            )[0]
            
            preds = np.squeeze(np.array(preds))
            for pred in preds:
                if pred <= 0.5:
                    test_pred.append(0)
                else:
                    test_pred.append(1)
                test_prob.append(pred)

        return test_pred, test_prob

    
if __name__ == "__main__":
    config = Config()
    train(config)
    print("Reading data...")
    train_reader, val_reader, test_reader = get_point_data(
        config.TRAIN_RATE, config.VAL_RATE, config.BATCH_SIZE, config.DOWN_SAMPLING, config.GOAL_FILE)
    inference(config, test_reader)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
