import pandas as pd
import numpy as np
import random
from collections import defaultdict

random.seed(42)

class Dataset(object):
    def __init__(self, data_tag):
        self.data_path = "../process_data/"
        self.utterance = self.file_loader(self.data_path + data_tag + "_utterance.txt")
        self.goal_type = self.file_loader(self.data_path + data_tag + "_type.txt")
        self.goal_entity = self.file_loader(self.data_path + data_tag + "_entity.txt")
        self.bot = self.file_loader(self.data_path + data_tag + "_bot.txt")
        self.label = self.file_loader(self.data_path + data_tag + "_label.txt")
        
        self.goal_type_graph = np.load(self.data_path + "graph_type_graph.npy")
        self.goal_entity_graph = np.load(self.data_path + "graph_entity_graph.npy")
        self.goal_type_dict = self.get_neighbour_dict(self.goal_type_graph)
        self.goal_entity_dict = self.get_neighbour_dict(self.goal_entity_graph)

    def binary_task_data(self):
        binary_utterance = list()
        binary_goal_type = list()
        bianry_label = list()
 
        for idx in range(len(self.utterance)):
            line_len = len(self.utterance[idx])
            for jdx in range(line_len):
                if self.bot[idx][jdx] == 1:
                    binary_utterance.append(self.utterance[idx][jdx - 1])
                    binary_goal_type.append(self.goal_type[idx][jdx - 1])
                    bianry_label.append(self.label[idx][jdx])
        return binary_utterance, binary_goal_type, bianry_label

    def next_goal_data(self, undersample=False):
        binary_goal_type = list()
        binary_final_goal_type = list()
        binary_goal_type_label = list()
        binary_goal_type_idx = list()

        binary_goal_entity = list()
        binary_final_goal_entity = list()
        binary_goal_entity_label = list()
        binary_goal_entity_idx = list()

        for idx in range(len(self.goal_type)):
            line_len = len(self.goal_type[idx])
            for jdx in range(line_len):
                if self.bot[idx][jdx] == 1:
                    pre_type_seq = self.goal_type[idx][:jdx]
                    pre_entity_seq = self.goal_entity[idx][:jdx]
                    if len(pre_type_seq) == 0 or len(pre_entity_seq) == 0:
                        continue
                    
                    pre_type_seq, pre_entity_seq = remove_repeat(pre_type_seq, pre_entity_seq)
                    for nb in self.goal_type_dict[pre_type_seq[-1]]:
                        binary_goal_type.append(pre_type_seq + [nb])
                        binary_goal_type_idx.append(idx)
                        binary_final_goal_type.append(self.goal_type[idx][-1])
                        if nb == self.goal_type[idx][jdx]:
                            binary_goal_type_label.append(1)
                        else:
                            binary_goal_type_label.append(0)
                    
                    cnt = 0
                    for nb in self.goal_entity_dict[pre_entity_seq[-1]]:
                        if nb == self.goal_entity[idx][jdx]:
                            binary_goal_entity.append(pre_entity_seq + [nb])
                            binary_goal_entity_label.append(1)
                            cnt += 1
                        else:
                            if cnt > 10 and random.random() > 0.2:
                                continue
                            binary_goal_entity.append(pre_entity_seq + [nb])
                            binary_goal_entity_label.append(0)
                            cnt += 1
                        binary_goal_entity_idx.append(idx)
                        binary_final_goal_entity.append(self.goal_entity[idx][-1])
                        
        return binary_goal_type, binary_goal_type_label, binary_goal_type_idx, binary_goal_entity, binary_goal_entity_label, binary_goal_entity_idx, binary_final_goal_type, binary_final_goal_entity
                    
    def get_neighbour_dict(self, graph):
        graph_dict = defaultdict(list)
        for idx, line in enumerate(graph):
            for jdx, num in enumerate(line):
                if num == 1:
                    graph_dict[idx].append(jdx)
        return graph_dict

    def file_loader(self, file_path):
        data = None
        with open(file_path, "r") as f:
            data = eval(f.read())
            f.close()
        return data

    
def remove_repeat(goal_seq, kg_seq):
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
    

def file_saver(file_path, obj):
    with open(file_path, "w") as f:
        f.write(str(obj))
        f.close()


def get_data(data_tag, undersample=False):
    data = Dataset(data_tag)
    binary_utterance, binary_goal_type, binary_label = data.binary_task_data()
    next_goal_type, next_goal_type_label, next_goal_type_idx, next_goal_entity, next_goal_entity_label, next_goal_entity_idx, final_goal_type, final_goal_entity = data.next_goal_data(undersample=undersample)
    
    print("Binary Jump Classification...")
    print("Sample Numebr: %d, Jump Number: %d, Jump Rate: %.2f" % (
        len(binary_utterance), np.sum(binary_label), float(np.sum(binary_label))/len(binary_utterance)))
    print("Next Goal Type Prediction...")
    print("Sample Numebr: %d, True Number: %d, True Rate: %.2f" % (
        len(next_goal_type), np.sum(next_goal_type_label), float(np.sum(next_goal_type_label))/len(next_goal_type)))
    print("Next Goal Entity Prediction...")
    print("Sample Numebr: %d, True Number: %d, True Rate: %.2f\n" % (
        len(next_goal_entity), np.sum(next_goal_entity_label), float(np.sum(next_goal_entity_label))/len(next_goal_entity)))

    save_path = "../train_data/"
    file_saver(save_path + data_tag + "_binary_utterance.txt", binary_utterance)
    file_saver(save_path + data_tag + "_binary_goal_type.txt", binary_goal_type)
    file_saver(save_path + data_tag + "_binary_label.txt", binary_label)
    file_saver(save_path + data_tag + "_next_goal_type.txt", next_goal_type)
    file_saver(save_path + data_tag + "_next_goal_type_idx.txt", next_goal_type_idx)
    file_saver(save_path + data_tag + "_next_goal_type_label.txt", next_goal_type_label)
    file_saver(save_path + data_tag + "_next_goal_entity.txt", next_goal_entity)
    file_saver(save_path + data_tag + "_next_goal_entity_idx.txt", next_goal_entity_idx)
    file_saver(save_path + data_tag + "_next_goal_entity_label.txt", next_goal_entity_label)
    file_saver(save_path + data_tag + "_final_goal_type.txt", final_goal_type)
    file_saver(save_path + data_tag + "_final_goal_entity.txt", final_goal_entity)



if __name__ == "__main__":
    train_data = get_data(data_tag="train", undersample=True)
    val_data = get_data(data_tag="val")