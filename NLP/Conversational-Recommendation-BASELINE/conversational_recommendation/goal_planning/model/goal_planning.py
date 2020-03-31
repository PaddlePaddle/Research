# -*- coding: utf-8 -*

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
from sklearn.metrics import confusion_matrix
import paddle.fluid.dygraph.base as base
from gensim import corpora
from gensim import models
from gensim import similarities
from collections import defaultdict
import jieba
import re
import json

reload(sys)
sys.setdefaultencoding('utf8')

class GoalPlanning(object):
    """Goal Planning"""
    def __init__(self):
        self.device = fluid.CPUPlace()
        self.binary_path = "../model_state/paddle_binary_lstm_v1.mdl"
        self.goal_type_path = "../model_state/paddle_astar_goal_v1.mdl"
        self.goal_entity_path = "../model_state/paddle_astar_kg_v1.mdl"
        
    def binary_inference(self, reader):
        """binary inference"""

        exe = fluid.Executor(self.device)
        inference_scope = fluid.core.Scope()
        with fluid.scope_guard(inference_scope):
            [inferencer, feed_target_names, fetch_targets] = fluid.io.load_inference_model(self.binary_path, exe)
            
            prediction = list()
            for data in reader():
                utterances = [d[0] for d in data]
                goals = [d[1] for d in data]
                
                base_shape = [[len(seq) for seq in utterances]]
                lod_text = fluid.create_lod_tensor(utterances, base_shape, self.device)
                feed_data = {
                    feed_target_names[0]: lod_text,
                    feed_target_names[1]: np.array(goals)
                }
                
                preds = exe.run(
                    inferencer,
                    feed=feed_data,
                    fetch_list=fetch_targets,
                    return_numpy=False
                )[0]
                preds = np.array(preds)
                for pred in preds:
                    if pred[0] > pred[1]:
                        prediction.append(0)
                    else:
                        prediction.append(1)
            return prediction
      
    def goal_type_inference(self, reader):
        """goal type inference"""
        exe = fluid.Executor(self.device)
        inference_scope = fluid.core.Scope()
        with fluid.scope_guard(inference_scope):
            [inferencer, feed_target_names, fetch_targets] = fluid.io.load_inference_model(self.goal_type_path, exe)

            prediction = list()
            for data in reader():
                goal_route = list()
                current = list()
                goal = list()
                for d in data:
                    goal_route.append(d[0])
                    current.append(d[1])
                    goal.append(d[2])
                    
                base_shape = [[len(seq) for seq in goal_route]]
                lod_seq = fluid.create_lod_tensor(goal_route, base_shape, self.device)

                new_data = {
                    feed_target_names[0]: lod_seq,
                    feed_target_names[1]: np.array(current),
                    feed_target_names[2]: np.array(goal)
                }

                pred = exe.run(
                    inferencer,
                    feed=new_data,
                    fetch_list=fetch_targets,
                    return_numpy=False
                )[0]

                pred = np.squeeze(np.array(pred))
                for p in pred:
                    prediction.append(p)
            return prediction
    
    def goal_entity_inference(self, reader):
        """goal entity inference"""
        exe = fluid.Executor(self.device)
        inference_scope = fluid.core.Scope()
        with fluid.scope_guard(inference_scope):
            [inferencer, feed_target_names, fetch_targets] = fluid.io.load_inference_model(self.goal_entity_path, exe)
            
            prediction = list()
            for data in reader():
                goal_route = list()
                current = list()
                goal = list()
                for d in data:
                    goal_route.append(d[0])
                    current.append(d[1])
                    goal.append(d[2])
                    
                base_shape = [[len(seq) for seq in goal_route]]
                lod_seq = fluid.create_lod_tensor(goal_route, base_shape, self.device)

                new_data = {
                    feed_target_names[0]: lod_seq,
                    feed_target_names[1]: np.array(current),
                    feed_target_names[2]: np.array(goal)
                }

                pred = exe.run(
                    inferencer,
                    feed=new_data,
                    fetch_list=fetch_targets,
                    return_numpy=False
                )[0]

                pred = np.squeeze(np.array(pred))
                for p in pred:
                    prediction.append(p)
            return prediction


def file_reader(file_name):
    """file_reader"""
    data = None
    with open(file_name, "r") as f:
        for line in f.readlines():
            data = eval(line)
        f.close()
    return data


def file_saver(file_name, file_obj):
    """file_saver"""
    with open(file_name, "w") as f:
        for fi in file_obj:
            f.write(json.dumps(fi, ensure_ascii=False) + '\n')


def reader_generater(datas):
    """reader_generater"""
    def reader():
        """reader"""
        for data in datas:
            yield data
    return reader


def remove_punctuation(line):
    """remove_punctuation"""
    print(line)
    #return re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9]', '', line)
    return re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+".decode('utf-8'), "".decode('utf-8'), line)

def word_replace(word):
    word = word.replace("问User", "问用户").replace("poi推荐", "兴趣点推荐").replace("\n", "")\
        .replace("『聊天 日期』", "『聊天日期』").replace(" 新闻", "新闻").replace("的新闻", "新闻").replace("说A好的幸福呢", "说好的幸福呢")
    word = remove_punctuation(word)
    return word

def remove_repeat(goal_seq, kg_seq):
    """remove_repeat"""
    assert len(goal_seq) == len(kg_seq)
    new_goal_seq, new_kg_seq = list(), list()
    for idx, (a, b) in enumerate(zip(goal_seq, kg_seq)):
        if idx > 0:
            if a == goal_seq[idx - 1] and b == kg_seq[idx - 1]:
                continue 
            else:
                new_goal_seq.append(a)
                new_kg_seq.append(b)
        else:
            new_goal_seq.append(a)
            new_kg_seq.append(b)
    
    return new_goal_seq, new_kg_seq


# def word_replace(word):
#     """word_replace"""
#     word = word.replace("问User", "问用户").replace("poi推荐", "兴趣点推荐").\
#     replace("\n", "").replace("『聊天 日期』", "『聊天日期』").replace(" 新闻", "新闻")
#     word = remove_punctuation(word)
#     return word


def evaluation(y_pred, y_true, flag="macro"):
    """evaluation"""
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, \
        average=flag), precision_score(y_true, y_pred, average=flag), f1_score(y_true, y_pred, average=flag)


def get_data(file_path):
    """get_data"""
    utterance = list()
    label = list()
    goal_type = list()
    goal_entity = list()

    with open(file_path, "r") as f:
        utt = list()
        lab = list()
        gtp = list()
        get = list()
        for line in f.readlines():
            if line == "\n":
                utt = [remove_punctuation(u) for u in utt]
                #utt = [u for u in utt]
                gtp = [remove_punctuation(word_replace(g)) for g in gtp]
                #gtp = [word_replace(g)  for g in gtp]
                get = [remove_punctuation(word_replace(g)) for g in get]
                #get = [word_replace(g) for g in get]
                print("utt", utt, type(utt))
                print("utt[:]", utt[:], type(utt[:]))
                utterance.append(utt[:])
                label.append(lab[:])
                goal_type.append(gtp[:])
                goal_entity.append(get[:])

                utt = list()
                lab = list()
                gtp = list()
                get = list()
            else:
                line = line.split("\t")
                if line[0] == "":
                    if line[2] == "再见":
                        utt.append("再见")
                    if line[2] == "音乐推荐":
                        utt.append("给你推荐一首歌吧")
                else:
                    utt.append(line[0])

                if line[1] == "":
                    lab.append(int(line[2]))
                    gtp.append(line[3])
                    if line[4] == "":
                        get.append(line[3])
                    else:
                        get.append(line[4])
                else:
                    lab.append(int(line[1]))
                    gtp.append(line[2])
                    if line[3] == "":
                        get.append(line[2])
                    else:
                        get.append(line[3])
        f.close()
    
    return utterance, label, goal_type, goal_entity


def get_name_dict(goal_dict_path, kg_dict_path):
    """get_name_dict"""
    goal_idx_dict = file_reader(goal_dict_path)
    kg_idx_dict = file_reader(kg_dict_path)
    idx_goal_dict = dict()
    idx_kg_dict = dict()
    
    for k, v in goal_idx_dict.items():
        idx_goal_dict[v] = k
    
    for k, v in kg_idx_dict.items():
        idx_kg_dict[v] = k
        
    return goal_idx_dict, idx_goal_dict, kg_idx_dict, idx_kg_dict 
 

def get_neighbour_dict(goal_neighbour_path, kg_neighbour_path):
    """get_neighbour_dict"""
    goal_neighbour_dict = file_reader(goal_neighbour_path)
    kg_neighbour_dict = file_reader(kg_neighbour_path)
    return goal_neighbour_dict, kg_neighbour_dict  


def data_process():
    """data_process"""
    utterance, label, goal_type, goal_entity = get_data("../origin_data/train.txt")
    word_dict = file_reader("../process_data/word_dict.txt")

    UNK = word_dict["UNK"]
    final_data = list()
    binary_utterance, binary_goal_type = list(), list()
    for idx in range(len(utterance)):
        gr, kr = remove_repeat(goal_type[idx], goal_entity[idx])
        gr = [word_replace(word) for word in gr]
        kr = [word_replace(word) for word in kr]
        
        for jdx in range(1, len(utterance[idx])):
            record = dict()
            record["session_idx"] = idx
            record["utterance_idx"] = jdx
            utt = re.sub(r"\[\d*\]", "", utterance[idx][jdx-1])
            utt = re.sub(r'[~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}]+', "", utt)
            utt = [word_dict.get(word, UNK) for word in utt.strip().split(" ") if word is not ""]

            record["utterance"] = utt
            record["label_truth"] = label[idx][jdx]
            record["final_goal_type"] = gr[-2]
            record["final_goal_entity"] = kr[-2]
            
            record["goal_type_route"] = goal_type[idx][:jdx + 1]
            record["goal_entity_route"] = goal_entity[idx][:jdx + 1]

            
            record["current_goal_type"] = goal_type[idx][jdx]
            record["current_goal_entity"] = goal_entity[idx][jdx]
            if jdx == 0:
                record["previous_goal_type"] = goal_type[idx][jdx]
                record["previous_goal_entity"] = goal_entity[idx][jdx]
            else:
                record["previous_goal_type"] = goal_type[idx][jdx - 1]
                record["previous_goal_entity"] = goal_entity[idx][jdx - 1]
            
            if record["label_truth"] == 1 and jdx != 0:
                ngr, nkr = remove_repeat(record["goal_type_route"], record["goal_entity_route"])
                if len(ngr) == len(gr):
                    record["next_goal_type"] = ngr[-1]
                    record["next_goal_entity"] = nkr[-1]
                else:
                    record["next_goal_type"] = gr[len(ngr)]
                    record["next_goal_entity"] = kr[len(nkr)]
            else:
                record["next_goal_type"] = goal_type[idx][jdx]
                record["next_goal_entity"] = goal_entity[idx][jdx]
            
            binary_utterance.append(utterance)
            binary_goal_type.append(record["previous_goal_type"])
            final_data.append(record)
    
    return final_data, binary_utterance, binary_goal_type


def get_argmax_pred(a, b):
    """get_argmax_pred"""
    group_data = list()
    
    idx = 0
    while idx < len(a):
        tmp_pred = list()
        tmp_type = list()
        tmp_idx = list()
        
        previous = a[idx][3]
        while idx < len(a) and a[idx][3] == previous:
            tmp_pred.append(b[idx])
            tmp_type.append(a[idx][1])
            tmp_idx.append(a[idx][3])
            idx += 1
            
        group_data.append([tmp_pred, tmp_type, tmp_idx])
        
#     for item in group_data:
#         print(item)
        
    data_pred = list()
    for item in group_data:
        pred_type = None
        max_pred = -1
        for a, b, c in zip(item[0], item[1], item[2]):
            if a > max_pred:
                max_pred = a
                pred_type = b
                
        if pred_type != None:
            data_pred.append((pred_type, c))
    return data_pred


def inference():
    """inference"""
    goal_idx_dict, idx_goal_dict, kg_idx_dict, idx_kg_dict = get_name_dict(
        "../process_data/goal_type_dict.txt", "../process_data/goal_entity_dict.txt")
    goal_neighbour_dict, kg_neighbour_dict = get_neighbour_dict(
        "../process_data/goal_type_neighbour.txt", "../process_data/goal_entity_neighbour.txt")
    # binary_utterance = file_reader("../train_data/train_binary_utterance.txt")
    # binary_goal_input = file_reader("../train_data/train_binary_goal_type.txt")
    
    data, binary_utterance, binary_goal_input = data_process()
    # binary_goal_input = [goal_idx_dict[t] for t in binary_goal_input]
    binary_goal_input = []
    for t in binary_goal_input:
        if t in goal_idx_dict:
            binary_goal_input.append(goal_idx_dict[t])
    model = GoalPlanning()
    print(len(data))
    
    binary_input = [[a, b] for a, b in zip(binary_utterance, binary_goal_input)]
    binary_input = paddle.batch(reader_generater(binary_input), batch_size=256)
    jump_flag_list = model.binary_inference(binary_input)

    goal_route_data = list()
    kg_route_data = list()
    next_goal_true = list()
    next_kg_true = list()
    for idx in range(len(data)):
        try:
            if jump_flag_list[idx] == 1 and data[idx]["utterance_idx"] != 0:
                goal_route = [goal_idx_dict[g] for g in data[idx]["goal_type_route"]]
                kg_route = [kg_idx_dict[k] for k in data[idx]["goal_entity_route"]]
                goal_route, kg_route = remove_repeat(goal_route, kg_route)

                final_goal_type = goal_idx_dict[data[idx]["final_goal_type"]]
                for nb in goal_neighbour_dict[goal_route[-1]]:
                    goal_route_data.append([goal_route + [nb], nb, final_goal_type, idx, \
                        goal_idx_dict[data[idx]["next_goal_type"]]])
                next_goal_true.append(goal_idx_dict[data[idx]["next_goal_type"]])

                final_goal_entity = kg_idx_dict[data[idx]["final_goal_entity"]]
                for nb in kg_neighbour_dict[kg_route[-1]]:
                    kg_route_data.append([kg_route + [nb], nb, final_goal_entity, idx, \
                        kg_idx_dict[data[idx]["next_goal_entity"]]])
                next_kg_true.append(kg_idx_dict[data[idx]["next_goal_entity"]])
        except:
            pass

    goal_route_data_ = paddle.batch(reader_generater(goal_route_data), batch_size=512)
    kg_route_data_ = paddle.batch(reader_generater(kg_route_data), batch_size=512)
    next_goal_pred = get_argmax_pred(goal_route_data, model.goal_type_inference(goal_route_data_))
    next_kg_pred = get_argmax_pred(kg_route_data, model.goal_entity_inference(kg_route_data_))
    print(len(next_goal_pred), len(next_goal_true))
    print(len(next_kg_pred), len(next_kg_true))
        
    goal_pred_dict = dict()
    kg_pred_dict = dict()
    
    for a, b in next_goal_pred:
        goal_pred_dict[b] = a
    for a, b in next_kg_pred:
        kg_pred_dict[b] = a
    
    for idx in range(len(data)):
        try:
            if jump_flag_list[idx] == 1 and data[idx]["utterance_idx"] != 0:
                data[idx]["next_goal_type_pred"] = idx_goal_dict[goal_pred_dict[idx]]
                data[idx]["next_goal_entity_pred"] = idx_kg_dict[kg_pred_dict[idx]]
            else:
                data[idx]["next_goal_type_pred"] = data[idx]["next_goal_type"]
                data[idx]["next_goal_entity_pred"] = data[idx]["next_goal_entity"]
        except:
            pass
  
    print(evaluation(list(goal_pred_dict.values()), next_goal_true))
    print(evaluation(list(kg_pred_dict.values()), next_kg_true))
    
    goal_type_pred, goal_type_true = list(), list()
    goal_entity_pred, goal_entity_true = list(), list()
    cnt = 0.
    for idx in range(len(data)):
        # if jump_flag_list[idx] == 1:
        try:
            goal_type_pred.append(goal_idx_dict[data[idx]["next_goal_type_pred"]])
            goal_type_true.append(goal_idx_dict[data[idx]["next_goal_type"]])
            goal_entity_pred.append(kg_idx_dict[data[idx]["next_goal_entity_pred"]])
            goal_entity_true.append(kg_idx_dict[data[idx]["next_goal_entity"]])
        
            if goal_type_pred[-1] == goal_type_true[-1] and goal_entity_pred[-1] == goal_entity_true[-1]:
                cnt += 1
        except:
            pass
        # print(goal_type_pred[-1], goal_type_true[-1], goal_entity_pred[-1], goal_entity_true[-1])
    print(evaluation(goal_type_pred, goal_type_true))
    print(evaluation(goal_entity_pred, goal_entity_true))
    print(cnt / len(data))
    
    file_saver("../train_data/goal_planning.txt", data)
    

if __name__ == "__main__":
    inference()




