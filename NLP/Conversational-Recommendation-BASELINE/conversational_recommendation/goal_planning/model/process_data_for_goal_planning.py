#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
File: get_data_for_goal_planning.py
"""

from __future__ import print_function
import sys
import json
import collections

import random


# reload(sys)
# sys.setdefaultencoding('utf8')

def is_start_with(str, start_str = ['[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]']):
    for s in start_str:
        if str.startswith(s):
            return 1
    return 0

def add_label(input_filename, output_filename):
    output_file = open(output_filename, 'w')
    for line in open(input_filename, 'r').readlines():
        data = json.loads(line.strip())
        conversation = data['conversation']
        label = []
        for c in conversation:
            label.append(is_start_with(c.strip()))
        data['label'] = label
        output_file.write(json.dumps(data, ensure_ascii = False) + '\n')
    output_file.close()

def process_session_data(input_filename, output_filename):
    bad_flag = ["参考知识"]
    flag = ["再见", "问天气", "问时间", "天气信息推送"]
    flag1 = ["关于明星的聊天", "音乐推荐", "播放音乐", "美食推荐", "poi推荐", "电影推荐", "音乐点播", "问日期", "新闻推荐", "新闻点播", "", "", ""]
    flag2 = ["问答"]
    flag3 = ["寒暄"]

    # user_profile key
    p_r_key = ["拒绝"]
    p_p_key = ["喜欢的电影", "喜欢的明星", "喜欢的poi", "喜欢的音乐", "喜欢的新闻"]
    p_a_key = ["同意的新闻", "同意的音乐", "同意的美食", "同意的poi", "同意的电影"]
    p_key = ["接受的电影", "接受的音乐", "没有接受的电影", "没有接受的音乐"]
    list_key = ["同意的新闻", "没有接受的音乐", "接受的电影", "喜欢的明星", "接受的音乐", "没有接受的电影", "喜欢的新闻"]
    
    all_flag = bad_flag + flag2 + flag + flag1 
    output_file = open(output_filename, 'w')
    user_profile_key_result = set()
    for line in open(input_filename, 'r'):
        flag_flag = 0
        entity_level_goal = ""
        count = 1
        data = json.loads(line.strip())
        situation = data['situation']
        conversation = data['conversation']
        goals = data['goal']
        label = data["label"]
        kg = data["knowledge"]
        user_profile = data["user_profile"]
        goals = goals.split('-->')
        current_entity_goal = ""
        used_kg_entity = set()
        for (s, r, o) in kg:
            if r != "适合吃" and s != '聊天':
                used_kg_entity.add(s)
        used_kg_entity = list(used_kg_entity)
        profile_entity_list = set()
        for key in user_profile:
            user_profile_key_result.add(key)
        for key in user_profile:
            keyword_list = p_p_key + p_a_key + p_r_key
            if key not in keyword_list:
                continue
            tmp_entity = user_profile[key]
            if isinstance(tmp_entity, list):
                for k in tmp_entity:
                    profile_entity_list.add(k.strip())
            else:
                profile_entity_list.add(tmp_entity.strip())
        profile_entity_list = list(profile_entity_list)
        if len(goals) != sum(label):
            continue
        first_goal = goals[0].strip().split(']', 1)[1].split('(', 1)[0]
        for i in range(len(label)):
            if first_goal == '寒暄':
                if i % 2 == 0:
                    dialog_flag = 'Bot'
                else:
                    dialog_flag = 'User'
            else:
                if i % 2 != 0:
                    dialog_flag = 'Bot'
                else:
                    dialog_flag = 'User'

            if label[i] == 1:
                current_goal = goals[count - 1].split('[{0}]'.format(count))[-1]
                domain = current_goal.split('(', 1)[0]
                if "『" and "』" not in current_goal:    #flag
                    output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + domain + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                    current_entity_goal = domain
                else:
                    if domain in flag1:
                        tmp = current_goal.split("『", 1)[-1].split("』", 1)[0]
                        if domain == "问日期":
                            output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + domain + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                            current_entity_goal = domain
                        elif "新闻" in domain:
                            output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + tmp + " 新闻" + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                            current_entity_goal = tmp + " 新闻"
                        else:
                            output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + tmp + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                            current_entity_goal = tmp
                    elif domain in flag2:
                        tmp1 = current_goal.split("『", 1)[-1].split("』", 1)[0]
                        tmp2 = current_goal.split("『", -1)[-1].split("』", -1)[0]
                        if tmp1 not in bad_flag:
                            if flag_flag == 0:
                                output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + tmp1 + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                                current_entity_goal = tmp2
                                flag_flag = 1
                            else:
                                output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + tmp2 + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                                current_entity_goal = tmp2
                                flag_flag = 0
                        else:
                            if flag_flag == 1:
                                output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + current_entity_goal + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                            else:
                                output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + tmp2 + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                                current_entity_goal = tmp2
                    else:
                        output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + domain + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
                count += 1
            else:
                output_file.write(conversation[i] + '\t' + str(label[i]) + '\t' + domain + '\t' + current_entity_goal + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
        output_file.write('\n')
    output_file.close()

def process_sample_data(input_filename, output_filename):
    bad_flag = ["参考知识"]
    flag = ["再见", "问天气", "问时间", "天气信息推送"]
    flag1 = ["关于明星的聊天", "音乐推荐", "播放音乐", "美食推荐", "poi推荐", "电影推荐", "音乐点播", "问日期", "新闻推荐", "新闻点播", "", "", ""]
    flag2 = ["问答", "新闻 点播", "音乐 点播"]
    flag3 = ["问 User 爱好", "天气 信息 推送"]
    flag4 = ["新闻 推荐"]

    # user_profile key
    p_r_key = ["拒绝"]
    p_p_key = ["喜欢的电影", "喜欢的明星", "喜欢的poi", "喜欢的音乐", "喜欢的新闻"]
    p_a_key = ["同意的新闻", "同意的音乐", "同意的美食", "同意的poi", "同意的电影"]
    p_key = ["接受的电影", "接受的音乐", "没有接受的电影", "没有接受的音乐"]
    list_key = ["同意的新闻", "没有接受的音乐", "接受的电影", "喜欢的明星", "接受的音乐", "没有接受的电影", "喜欢的新闻"]
    
    all_flag = bad_flag + flag2 + flag + flag1 
    output_file = open(output_filename, 'w')
    user_profile_key_result = set()
    next_utterance = ''
    utterance_flag = 0
    for line in open(input_filename, 'r'):
        flag_flag = 0
        entity_level_goal = ""
        count = 1
        data = json.loads(line.strip())
        situation = data['situation']
        conversation = '\001'.join(data['history'])
        goals = data['goal']
        label = 0
        kg = data["knowledge"]
        user_profile = data["user_profile"]
        goals = goals.split('-->')
        current_entity_goal = ""
        used_kg_entity = set()
        for (s, r, o) in kg:
            if r != "适合吃" and s != '聊天':
                used_kg_entity.add(s)
        used_kg_entity = list(used_kg_entity)
        profile_entity_list = set()
        for key in user_profile:
            user_profile_key_result.add(key)
        for key in user_profile:
            keyword_list = p_p_key + p_a_key + p_r_key
            if key not in keyword_list:
                continue
            tmp_entity = user_profile[key]
            if isinstance(tmp_entity, list):
                for k in tmp_entity:
                    profile_entity_list.add(k.strip())
            else:
                profile_entity_list.add(tmp_entity.strip())
        profile_entity_list = list(profile_entity_list)

        dialog_flag = 'Bot'
        first_goal = goals[0].strip().split(']', 1)[1].split('(', 1)[0].strip()
        if '......' in data['goal']:
            final_goal = goals[2].strip().split(']', 1)[1].split('(', 1)[0].strip()
        else:
            final_goal = goals[1].strip().split(']', 1)[1].split('(', 1)[0].strip()
        try:
            first_utterance = data['history'][0]
        except:
            first_utterance = ''
        if first_utterance != next_utterance and first_goal != '寒暄':
            if utterance_flag != 0:
                output_file.write('\n')
            utterance_flag = 1
            next_utterance = first_utterance
        if first_goal in flag2:
            if '『 参考 知识 』' in data['goal']:
                first_goal_topic = data['goal'].strip().split('『 参考 知识 』')[-1].split('『 ', 1)[-1].split('』', 1)[0].strip()
            else:
                first_goal_topic = data['goal'].strip().split('『 ', 1)[-1].split('』', 1)[0].strip()
        else:
            first_goal_topic = first_goal
        if final_goal in flag3:
            final_goal_topic = final_goal
        else:
            final_goal_topic = data['goal'].split(final_goal)[-1].split('『 ', 1)[-1].split('』', 1)[0].strip()
            if final_goal in flag4:
                final_goal_topic += ' 的 新闻'
        output_file.write(conversation + '\t' + str(label) + '\t' + first_goal + '\t' + first_goal_topic + '\t' + final_goal + '\t' + final_goal_topic + '\t' + str(kg) + '\t' + str(user_profile) + '\t' + dialog_flag + '\n')
    output_file.close()


if __name__ == '__main__':
    add_label('../origin_data/resource/train.txt', '../origin_data/train_add_label.txt')
    add_label('../origin_data/resource/dev.txt', '../origin_data/dev_add_label.txt')
    process_session_data('../origin_data/train_add_label.txt', '../origin_data/train.txt')
    process_session_data('../origin_data/dev_add_label.txt', '../origin_data/dev.txt')
    process_sample_data('../origin_data/resource/test_1.txt', '../origin_data/test.txt')


