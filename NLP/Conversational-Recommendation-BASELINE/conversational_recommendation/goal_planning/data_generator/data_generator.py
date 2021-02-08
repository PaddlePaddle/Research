#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import re
from collections import defaultdict

def file_reader(file_path):
    utterance = list()
    label = list()
    goal_type = list()
    goal_entity = list()
    bot_flag = list()

    with open(file_path, "r") as f:
        utt = list()
        lab = list()
        gtp = list()
        get = list()
        bfl = list()
        for line in f.readlines():
            if line == "\n":
                utterance.append(utt)
                label.append(lab)
                goal_type.append(gtp)
                goal_entity.append(get)
                bot_flag.append(bfl)
                
                utt = list()
                lab = list()
                gtp = list()
                get = list()
                bfl = list()
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
                bfl.append(line[-1].replace("\n", ""))
        f.close()
    return utterance, goal_type, goal_entity, bot_flag, label

def file_loader(file_path):
    data = None
    with open(file_path, "r") as f:
        data = eval(f.read())
        f.close()
    return data

def file_saver(file_path, obj):
    with open(file_path, "w") as f:
        f.write(str(obj))
        f.close()

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

def word_replace(word):
    word = word.replace("问User", "问用户").replace("poi推荐", "兴趣点推荐").replace("\n", "").replace("『聊天 日期』", "『聊天日期』").replace(" 新闻", "新闻").replace("的新闻", "新闻")
    word = word.replace("说A好的幸福呢", "说好的幸福呢")
    word = remove_punctuation(word)
    return word

def remove_punctuation(line):
    line = re.sub("\[\d*\]", "", line)
    return re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9]', '', line)

def get_data_idx(utt, type, entity, bot, word_dict, type_dict, entity_dict, bot_dict):
    utt_idx = text_generator(word_dict, utt)
    type_idx = list()
    entity_idx = list()
    bot_idx = list()
    for idx in range(len(type)):
        type_idx.append([type_dict[word_replace(word.replace(" ", ""))] for word in type[idx]])
        entity_idx.append([entity_dict[word_replace(word.replace(" ", ""))] for word in entity[idx]])
        bot_idx.append([bot_dict[b] for b in bot[idx]])
    return utt_idx, type_idx, entity_idx, bot_idx

def get_graph(train_data, val_data, graph_len, item_dict=None, flag=False):
    graph = np.eye(graph_len, graph_len)

    for idx in range(len(train_data)):
        for jdx in range(len(train_data[idx]) - 1):
            graph[train_data[idx][jdx]][train_data[idx][jdx+1]] = 1

    for idx in range(len(val_data)):
        for jdx in range(len(val_data[idx]) - 1):
            graph[val_data[idx][jdx]][val_data[idx][jdx+1]] = 1

    if flag is True:
        all_star = file_loader("../origin_data/all_star.txt")
        for star in all_star:
            graph[item_dict[star]][item_dict[star+"新闻"]] = 1
            
        with open("../origin_data/final_star2movie.txt", "r") as f:
            for movie in f.readlines():
                star, movie_list = movie.split("\001")
                star = remove_punctuation(star)
                if star in all_star:
                    movie_list = [remove_punctuation(mv) for mv in movie_list.split("\t")]
                    for mv in movie_list:
                        graph[item_dict[star]][item_dict[mv]] = 1
                        
        with open("../origin_data/singer2song_with_comment.txt", "r") as f:
            for music in f.readlines():
                star, music_list = music.split("\001")
                star = remove_punctuation(star)
                if star in all_star:
                    music_list = [remove_punctuation(mc) for mc in music_list.split("\t")]
                    for mc in music_list:
                        graph[item_dict[star]][item_dict[mc]] = 1

        with open("../origin_data/food_kg_human_filter.json", "r") as f:
            for line in f.readlines():
                line = eval(line)
                city = remove_punctuation(line["city"])
                poi = remove_punctuation(line["shopName"])
                food = remove_punctuation(line["name"])
                graph[item_dict[city]][item_dict[poi]] = 1
                graph[item_dict[city]][item_dict[food]] = 1
                graph[item_dict[poi]][item_dict[food]] = 1
                graph[item_dict[food]][item_dict[poi]] = 1
    return graph

def get_word_dict(utterances):
    stop_words = list()
    with open("../origin_data/stop_words.txt", "r") as f:
        for line in f.readlines():
            stop_words.append(line.replace("\n", ""))

    word_set = set()
    for utterance in utterances:
        for u in utterance:
            u = re.sub("\[\d*\]", "", u)
            u = re.sub(r'[~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}]+', "", u)
            for word in u.strip().split(" "):
                if word is not "":
                    word_set.add(word)
    
    word_dict = dict()
    for idx, word in enumerate(word_set):
        word_dict[word] = idx
    word_dict["UNK"] = len(word_dict)
    return word_dict


def text_generator(word_dict, documents):
    texts_idx = list()
    UNK = word_dict["UNK"]
    for doc in documents:
        doc_idx = list()
        for line in doc:
            line = re.sub("\[\d*\]", "", line)
            line = re.sub(r'[~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}]+', "", line)
            line_idx = [word_dict.get(word, UNK) for word in line.strip().split(" ") if word is not ""]
            doc_idx.append(line_idx)
        texts_idx.append(doc_idx)
    return texts_idx


def get_test_data(word_dict, goal_type_dict, goal_entity_dict):
    binary_utterance = list()
    binary_label = list()
    binary_goal_type = list()

    next_goal_type = list()
    next_goal_entity = list()
    next_final_goal_type = list()
    next_final_goal_entity = list()
    next_goal_type_idx = list()
    next_goal_entity_idx = list()

    UNK = word_dict["UNK"]
    with open("../origin_data/test.txt", "r") as f:
        for idx, line in enumerate(f.readlines()):
            if line == "\n":
                continue
            line = line.split("\t")
            utterance = line[0].split("\001")[-1]
            if utterance == "":
                continue
            utterance = re.sub("\[\d*\]", "", utterance)
            utterance = re.sub(r'[~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}]+', "", utterance)
            utterance = [word_dict.get(word, UNK) for word in utterance.strip().split(" ") if word is not ""]

            binary_utterance.append(utterance)
            binary_label.append(line[1])
            binary_goal_type.append(goal_type_dict[word_replace(line[2].replace(" ", ""))])

            next_goal_type.append([goal_type_dict[word_replace(line[2].replace(" ", ""))]])
            next_goal_entity.append([goal_entity_dict[word_replace(line[3].replace(" ", ""))]])
            next_final_goal_type.append([goal_type_dict[word_replace(line[4].replace(" ", ""))]])
            next_final_goal_entity.append([goal_entity_dict[word_replace(line[5].replace(" ", ""))]])
            next_goal_type_idx.append(idx)
            next_goal_entity_idx.append(idx)
        
        f.close()
    save_path = "../train_data/"
    data_tag = "test"
    file_saver(save_path + data_tag + "_binary_utterance.txt", binary_utterance)
    file_saver(save_path + data_tag + "_binary_goal_type.txt", binary_goal_type)
    file_saver(save_path + data_tag + "_binary_label.txt", binary_label)
    file_saver(save_path + data_tag + "_next_goal_type.txt", next_goal_type)
    file_saver(save_path + data_tag + "_next_goal_type_idx.txt", next_goal_type_idx)
    file_saver(save_path + data_tag + "_next_goal_entity.txt", next_goal_entity)
    file_saver(save_path + data_tag + "_next_goal_entity_idx.txt", next_goal_entity_idx)
    file_saver(save_path + data_tag + "_final_goal_type.txt", next_final_goal_type)
    file_saver(save_path + data_tag + "_final_goal_entity.txt", next_final_goal_entity)
    

if __name__ == "__main__":
    origin_data_path = "../origin_data/"
    train_data_path = origin_data_path + "train.txt"
    val_data_path = origin_data_path + "dev.txt"
    test_data_path = origin_data_path + "test.txt"
    train_utt, train_type, train_entity, train_bot, train_label = file_reader(train_data_path)
    val_utt, val_type, val_entity, val_bot, val_label = file_reader(val_data_path)

    word_dict = get_word_dict(train_utt)
    print("word_dict_size: %d" % (len(word_dict)))
    all_goal_type = file_loader(origin_data_path + "all_goal_type.txt")
    all_goal_entity = file_loader(origin_data_path + "all_goal_entity.txt")
    all_goal_entity = set.union(all_goal_type, all_goal_entity)
    print(len(all_goal_type), len(all_goal_entity))

    file_saver("../process_data/word_dict.txt", word_dict)
    all_goal_type_dict = dict()
    all_goal_entity_dict = dict()
    for idx, item in enumerate(all_goal_type):
        all_goal_type_dict[item] = idx
    cnt = 0
    for idx, item in enumerate(all_goal_entity):
        if all_goal_entity_dict.get(item) != None:
            continue
        all_goal_entity_dict[item] = cnt
        cnt += 1
    bot_dict = {"Bot": 1, "User": 0}

    # get_test_data(word_dict, all_goal_type_dict, all_goal_entity_dict)
    # train_utt, train_type, train_entity, train_bot, train_label = file_reader(train_data_path)
    # val_utt, val_type, val_entity, val_bot, val_label = file_reader(val_data_path)
    
    # train_utt_idx, train_type_idx, train_entity_idx, train_bot_idx = get_data_idx(
    #     train_utt, train_type, train_entity, train_bot, word_dict, all_goal_type_dict, all_goal_entity_dict, bot_dict)
    # val_utt_idx, val_type_idx, val_entity_idx, val_bot_idx = get_data_idx(
    #     val_utt, val_type, val_entity, val_bot, word_dict, all_goal_type_dict, all_goal_entity_dict, bot_dict)

    # goal_type_graph = get_graph(train_type_idx, val_type_idx, len(all_goal_type_dict))
    # goal_entity_graph = get_graph(
    #     train_entity_idx, val_entity_idx, len(all_goal_entity_dict), item_dict=all_goal_entity_dict, flag=True)

    # goal_type_neighbour = dict()
    # goal_entity_neighbour = dict()
    # for idx, line in enumerate(goal_type_graph):
    #     goal_type_neighbour[idx] = list()
    #     for jdx, num in enumerate(goal_type_graph[idx]):
    #         goal_type_neighbour[idx].append(jdx)
    # for idx, line in enumerate(goal_entity_graph):
    #     goal_entity_neighbour[idx] = list()
    #     for jdx, num in enumerate(goal_entity_graph[idx]):
    #         goal_entity_neighbour[idx].append(jdx)

    # file_saver("../process_data/goal_type_neighbour.txt", goal_type_neighbour)
    # file_saver("../process_data/goal_entity_neighbour.txt", goal_entity_neighbour)

    # file_saver("../process_data/train_utterance.txt", train_utt_idx)
    # file_saver("../process_data/train_type.txt", train_type_idx)
    # file_saver("../process_data/train_entity.txt", train_entity_idx)
    # file_saver("../process_data/train_bot.txt", train_bot_idx)
    # file_saver("../process_data/train_label.txt", train_label)
    
    # file_saver("../process_data/val_utterance.txt", val_utt_idx)
    # file_saver("../process_data/val_type.txt", val_type_idx)
    # file_saver("../process_data/val_entity.txt", val_entity_idx)
    # file_saver("../process_data/val_bot.txt", val_bot_idx)
    # file_saver("../process_data/val_label.txt", val_label)

    # file_saver("../process_data/goal_type_dict.txt", all_goal_type_dict)
    # file_saver("../process_data/goal_entity_dict.txt", all_goal_entity_dict)

    # np.save("../process_data/graph_type_graph", goal_type_graph)
    # np.save("../process_data/graph_entity_graph", goal_entity_graph)




