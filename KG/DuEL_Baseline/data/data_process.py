# -*- coding: utf-8 -*-
########################################################
# Copyright (c) 2020, Baidu Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
########################################################
import json
import os
import pickle
import random
import sys

reload(sys)
sys.setdefaultencoding('utf8')


class GenerateCand(object):
    """gnerate candidate entity id for mention"""

    def __init__(self):
        self.cand_dic = {}

    def interface(self, file_path):
        """interface"""
        for line in open('./basic_data/' + file_path):
            line_json = json.loads(line.strip())
            mention_list = line_json.get('alias')
            subject_name = line_json.get('subject')
            mention_list.append(subject_name)
            subject_id = line_json.get('subject_id')
            subject_type = line_json.get('type')
            ent_desc = ''
            for item in line_json.get('data'):
                ent_desc += item.get('predicate') + ':' + \
                    item.get('object') + ';'
            for mention in mention_list:
                if mention in self.cand_dic:
                    self.cand_dic[mention]['iid_list'].append(subject_id)
                    self.cand_dic[mention]['iid_list'] = list(
                        set(self.cand_dic[mention]['iid_list']),
                    )
                    self.cand_dic[mention][subject_id] = {}
                    self.cand_dic[mention][subject_id]['ent_desc'] = ent_desc
                    self.cand_dic[mention][subject_id]['type'] = subject_type
                else:
                    self.cand_dic[mention] = {}
                    self.cand_dic[mention]['iid_list'] = [subject_id]
                    self.cand_dic[mention][subject_id] = {}
                    self.cand_dic[mention][subject_id]['ent_desc'] = ent_desc
                    self.cand_dic[mention][subject_id]['type'] = subject_type
        pickle.dump(
            self.cand_dic, open(
                './generated/' + file_path + '.cand.pkl', 'wb',
            ),
        )
        return self.cand_dic

    def load_data(self, file_path):
        """load generated candidate dict"""
        self.cand_dic = pickle.load(
            open('./generated/' + file_path + '.cand.pkl', 'rb'),
        )
        return self.cand_dic


class DataProcess(object):
    """data process"""

    def __init__(self):
        self.type_label_map = {}
        self.type_num = 0
        self.type_label_map_reverse = {}

    def interface(self, file_path, cand_dic, is_train=True):
        """interface"""
        out_file = open(
            './generated/' +
            file_path.replace('.json', '') + '.txt', 'wb',
        )
        out_file.write(
            'qid\tqid_text\ttext_a\ttext_b\ttext_c\tlabel\ttype\tent_id_b\tent_id_c\n',
        )
        qid_mention = 1
        for line in open('./basic_data/' + file_path):
            line_json = json.loads(line.strip())
            text_id = line_json.get('text_id')
            query = line_json.get('text')
            mention_data = line_json.get('mention_data')
            for item in mention_data:
                mention = item.get('mention')
                if is_train:
                    kb_id = item.get('kb_id')
                    offset = item.get('offset')
                    if mention not in cand_dic:
                        continue
                    cand = cand_dic[mention]
                    iid_list = cand['iid_list']
                    if 'NIL' in kb_id:
                        golden_desc = mention
                        if '|' in kb_id:
                            kb_id = kb_id.split('|')[0]
                        golden_type = kb_id.replace('NIL_', '')
                        kb_id = 'NIL'
                    else:
                        golden_desc = cand[kb_id]['ent_desc']
                        golden_desc = golden_desc.replace('\015', '')
                        golden_type = cand[kb_id]['type']
                        if '|' in golden_type:
                            golden_type = golden_type.split('|')[0]
                    if golden_type.decode('utf8') not in self.type_label_map:
                        self.type_label_map[golden_type.decode(
                            'utf8',
                        )] = self.type_num
                        self.type_num += 1
                else:
                    if mention not in cand_dic:
                        cand = {}
                        iid_list = []
                    else:
                        cand = cand_dic[mention]
                        iid_list = cand['iid_list']
                cand['NIL'] = {}
                cand['NIL']['ent_desc'] = mention
                iid_list.append('NIL')
                iid_list = list(set(iid_list))

                for iid in iid_list:
                    tmp_desc = cand[iid]['ent_desc']
                    tmp_desc = tmp_desc.replace('\015', '')
                    if not is_train:
                        out_file.write(
                            str(qid_mention) + '\t' + text_id + '\t' + query + ';' + mention + '\t' +
                            tmp_desc + '\t' + tmp_desc + '\t' + 'null' + '\t' +
                            'null' + '\t' + iid + '\t' + iid + '\n',
                        )
                        continue
                    elif iid == kb_id:
                        continue
                    random_threshold = random.random()
                    if random_threshold > 0.5:
                        out_file.write(
                            str(qid_mention) + '\t' + text_id + '\t' + query + ';' + mention + '\t' +
                            golden_desc + '\t' + tmp_desc + '\t' + '1' + '\t' +
                            golden_type + '\t' + kb_id + '\t' + iid + '\n',
                        )
                    else:
                        out_file.write(
                            str(qid_mention) + '\t' + text_id + '\t' + query + ';' + mention + '\t' +
                            tmp_desc + '\t' + golden_desc + '\t' + '0' + '\t' +
                            golden_type + '\t' + iid + '\t' + kb_id + '\n',
                        )
                qid_mention += 1

        if is_train:
            for key, value in self.type_label_map.items():
                self.type_label_map_reverse[value] = key
            type_label_map_file = open('./generated/type_label_map.json', 'wb')
            type_label_map_reverse_file = open(
                './generated/type_label_map_reverse.json', 'wb',
            )
            type_label_map_file.write(json.dumps(
                self.type_label_map, ensure_ascii=False,
            ).encode('utf8'))
            type_label_map_reverse_file.write(json.dumps(
                self.type_label_map_reverse, ensure_ascii=False,
            ).encode('utf8'))


if __name__ == '__main__':
    generate_cand = GenerateCand()
    if os.path.exists('./genearted/cand.pkl'):
        cand_dic = generate_cand.load_data('kb.json')
    else:
        cand_dic = generate_cand.interface('kb.json')

    data_process = DataProcess()
    data_process.interface('train.json', cand_dic, is_train=True)
    # The dev.json parameter is_train is set to True during the training phase
    # and False during the prediction phase
    data_process.interface('dev.json', cand_dic, is_train=True)
    data_process.interface('test.json', cand_dic, is_train=False)
