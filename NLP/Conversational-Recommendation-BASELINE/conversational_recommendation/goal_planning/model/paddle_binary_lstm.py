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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


class Config(object):
    """Config"""
    def __init__(self):
        self.CLASS_SIZE = 2
        self.EMBED_SIZE = 128
        self.HIDDEN_SIZE = 128
        self.STACKED_NUM = 3
        self.GOAL_TYPE_SIZE = 21
        self.WORD_DICT_SIZE = 14816
        self.TRAIN_RATE = 0.7
        self.VAL_RATE = 0.15
        self.SHUFFLE = False
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.001
        self.NUM_EPOCH = 50
        self.DEVICE = fluid.CPUPlace()
        self.SAVE_PATH = "../model_state/paddle_binary_lstm_v1.mdl"

      
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


def data_zip(utterance, goal_input, label):
    all_data = list()
    for a, b, c in zip(utterance, goal_input, label):
        all_data.append([a, b, c])
    return all_data


def get_binary_data(train_rate, val_rate, batch_size):
    """get_binary_data"""
    train_utterance = file_reader("../train_data/train_binary_utterance.txt")
    train_goal_input = file_reader("../train_data/train_binary_goal_type.txt")
    train_label = file_reader("../train_data/train_binary_label.txt")
    
    val_utterance = file_reader("../train_data/val_binary_utterance.txt")
    val_goal_input = file_reader("../train_data/val_binary_goal_type.txt")
    val_label = file_reader("../train_data/val_binary_label.txt")
    
    test_utterance = file_reader("../train_data/test_binary_utterance.txt")
    test_goal_input = file_reader("../train_data/test_binary_goal_type.txt")
    test_label = file_reader("../train_data/test_binary_label.txt")
    
    train_data = data_zip(train_utterance, train_goal_input, train_label)
    val_data = data_zip(val_utterance, val_goal_input, val_label)
    test_data = data_zip(test_utterance, test_goal_input, test_label)
    
    train_len = len(train_data)
    val_len = len(val_data)
    data_len = train_len + val_len + len(test_data)
    
    print("train_len:%d, val_len:%d, test_len:%d" % (len(train_data), len(val_data), len(test_data)))
    train_reader = paddle.batch(reader_generater(train_data), batch_size=batch_size)
    val_reader = paddle.batch(reader_generater(val_data), batch_size=batch_size)
    test_reader = reader_generater(test_data)
    return train_reader, val_reader, test_reader


def lstm_net(text, goal, config):
    """lstm_net"""
    input_dim = config.WORD_DICT_SIZE
    class_dim = config.CLASS_SIZE
    goal_type_dim = config.GOAL_TYPE_SIZE
    embed_dim = config.EMBED_SIZE
    hidden_dim = config.HIDDEN_SIZE
    stacked_num = config.STACKED_NUM
    
    text_embed = fluid.embedding(input=text, size=[input_dim, embed_dim], is_sparse=True)
    
    fc1 = fluid.layers.fc(input=text_embed, size=hidden_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hidden_dim)
    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hidden_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hidden_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')
    
    goal_embed = fluid.embedding(input=goal, size=[goal_type_dim, embed_dim], is_sparse=True)
    
    pred_fc_1 = fluid.layers.fc(input=[fc_last, lstm_last, goal_embed], size=64, act="relu")
    prediction = fluid.layers.fc(input=pred_fc_1, size=class_dim, act="softmax")
    return prediction 


def inference_program(config):
    """inference_program"""
    text = fluid.data(name="text", shape=[None, 1], dtype="int64", lod_level=1)
    goal = fluid.data(name="goal", shape=[None], dtype="int64")
    net = lstm_net(text, goal, config)
    return net


def train_program(prediction):
    """train_program"""
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]


def evaluation(y_pred, y_true, flag="macro"):
    """evaluation"""
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, average=flag), \
    precision_score(y_true, y_pred, average=flag), f1_score(y_true, y_pred, average=flag)


def optimizer_func(learning_rate):
    """optimizer_func"""
    return fluid.optimizer.Adagrad(learning_rate=learning_rate)


def train(config):
    """train"""
    print("Reading data....")
    train_reader, val_reader, test_reader = get_binary_data(config.TRAIN_RATE, \
        config.VAL_RATE, config.BATCH_SIZE)
    
    feed_order = ["text", "goal", "label"]
    main_program = fluid.default_main_program()
    star_program = fluid.default_startup_program()
    main_program.random_seed = 90
    star_program.random_seed = 90
    
    prediction = inference_program(config)
    train_func_outputs = train_program(prediction)
    avg_cost = train_func_outputs[0]
    
    val_program = main_program.clone(for_test=True)
    
    sgd_optimizer = optimizer_func(config.LEARNING_RATE)
    sgd_optimizer.minimize(avg_cost)
    exe = fluid.Executor(config.DEVICE)
    
    def val_loop(program, reader):
        """val_loop"""
        feed_var_list = [program.global_block().var(var_name) for var_name in feed_order]
        feeder_var = fluid.DataFeeder(feed_list=feed_var_list, place=config.DEVICE)
        val_exe = fluid.Executor(config.DEVICE)
        val_pred, val_true = list(), list()
        for val_data in reader():
            preds = val_exe.run(
                program=program,
                feed=feeder_var.feed(val_data),
                fetch_list=prediction
            )[0]
            for pred in preds:
                if pred[0] > pred[1]:
                    val_pred.append(0)
                else:
                    val_pred.append(1)
            for vd in val_data:
                val_true.append(vd[2])
        return evaluation(val_pred, val_true)
    
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
                    fetch_list=[var.name for var in train_func_outputs]
                )
                train_loss.append(np.array(metrics[0]))
                train_acc.append(np.array(metrics[1]))
                
            val_acc, val_rec, val_pre, val_f1 = val_loop(val_program, val_reader)
            print("epoch_id: %d, avg_train_loss: %.4f, avg_train_acc: %.4f, val_acc: %.4f, \
                val_rec: %.4f, val_pre: %.4f, val_f1: %.4f" % (
                epoch_id, np.mean(train_loss), np.mean(train_acc), val_acc, val_rec, val_pre, val_f1
            ))
            
            if val_acc > max_acc:
                max_acc = val_acc
                max_epoch = epoch_id
                fluid.io.save_inference_model(config.SAVE_PATH, ["text", "goal"], prediction, exe)
            break
        print("max_epoch: %d, max_acc: %.4f" % (max_epoch, max_acc))
        
    train_loop()
    infer(config, test_reader)


def infer(config, reader):
    """infer"""
    exe = fluid.Executor(config.DEVICE)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inferencer, feed_target_names, fetch_targets] = fluid.io.load_inference_model(config.SAVE_PATH, exe)

        infer_pred = list()
        for data in reader():
            if len(data[0]) == 0:
                continue
            lod_text = fluid.create_lod_tensor([data[0]], [[len(data[0])]], config.DEVICE)
            pred = exe.run(
                inferencer,
                feed={feed_target_names[0]: lod_text, feed_target_names[1]: np.array([data[1]])},
                fetch_list=fetch_targets,
                return_numpy=False
            )[0]
            pred = np.array(pred)[0]
            if pred[0] > pred[1]:
                infer_pred.append(0)
            else:
                infer_pred.append(1)

        return infer_pred
        

if __name__ == "__main__":
    config = Config()
    train(config)



