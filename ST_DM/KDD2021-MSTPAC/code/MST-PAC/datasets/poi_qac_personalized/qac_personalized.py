#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################

"""
 Specify the brief poi_qac_personalized.py
"""
import os
import sys
import six
import re
import time
import numpy as np
import random
import datetime
import paddle.fluid as fluid

from datasets.base_dataset import BaseDataset
from utils.common_lib import convert_to_unicode

if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf-8')

base_rule = re.compile("[\1\2]")

class PoiQacPersonalized(BaseDataset):
    """
    PoiQacPersonalized dataset 
    """
    def __init__(self, flags):
        super(PoiQacPersonalized, self).__init__(flags)
        self.inited_dict = False

    def parse_context(self, inputs):
        """
        provide input context
        """

        """
        set inputs_kv: please set key as the same as layer.data.name

        notice:
        (1)
        If user defined "inputs key" is different from layer.data.name,
        the frame will rewrite "inputs key" with layer.data.name
        (2)
        The param "inputs" will be passed to user defined nets class through
        the nets class interface function : net(self, FLAGS, inputs), 
        """ 
        inputs['prefix_letter_id'] = fluid.layers.data(name="prefix_letter_id", shape=[1],
                dtype="int64", lod_level=1)
        if self._flags.prefix_word_id:
            inputs['prefix_word_id'] = fluid.layers.data(name="prefix_word_id", shape=[1],
                dtype="int64", lod_level=1)
        if self._flags.use_geohash:
            inputs['prefix_loc_geoid'] = fluid.layers.data(name="prefix_loc_geoid", shape=[40],
                dtype="int64", lod_level=0)

        inputs['pos_name_letter_id'] = fluid.layers.data(name="pos_name_letter_id", shape=[1],
                dtype="int64", lod_level=1)
        inputs['pos_addr_letter_id'] = fluid.layers.data(name="pos_addr_letter_id", shape=[1],
                dtype="int64", lod_level=1)
        if self._flags.poi_word_id:
            inputs['pos_name_word_id'] = fluid.layers.data(name="pos_name_word_id", shape=[1],
                dtype="int64", lod_level=1)
            inputs['pos_addr_word_id'] = fluid.layers.data(name="pos_addr_word_id", shape=[1],
                dtype="int64", lod_level=1)
        if self._flags.use_geohash:
            inputs['pos_loc_geoid'] = fluid.layers.data(name="pos_loc_geoid", shape=[40],
                dtype="int64", lod_level=0)

        if self.is_training:
            inputs['neg_name_letter_id'] = fluid.layers.data(name="neg_name_letter_id", shape=[1],
                    dtype="int64", lod_level=1)
            inputs['neg_addr_letter_id'] = fluid.layers.data(name="neg_addr_letter_id", shape=[1],
                    dtype="int64", lod_level=1)
            
            if self._flags.poi_word_id:
                inputs['neg_name_word_id'] = fluid.layers.data(name="neg_name_word_id", shape=[1],
                    dtype="int64", lod_level=1)
                inputs['neg_addr_word_id'] = fluid.layers.data(name="neg_addr_word_id", shape=[1],
                    dtype="int64", lod_level=1)
            if self._flags.use_geohash:
                inputs['neg_loc_geoid'] = fluid.layers.data(name="neg_loc_geoid", shape=[40],
                    dtype="int64", lod_level=0)
            
        else:
            #for predict label
            inputs['label'] = fluid.layers.data(name="label", shape=[1],
                dtype="int64", lod_level=0)
            inputs['qid'] = fluid.layers.data(name="qid", shape=[1],
                dtype="int64", lod_level=0)
            

        context = {"inputs": inputs}

        #set debug list, print info during training
        #context["debug_list"] = [key for key in inputs] 

        return context

    def _init_dict(self):
        """
            init dict
        """
        if self.inited_dict:
            return
        if self._flags.platform in ('local-gpu', 'pserver-gpu', 'slurm'):
            gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            self.place = fluid.CUDAPlace(gpu_id)
        else:
            self.place = fluid.CPUPlace()

        self.term_dict = {}
        if self._flags.qac_dict_path is not None:
            with open(self._flags.qac_dict_path, 'r') as f:
                for line in f:
                    term, term_id = line.strip('\r\n').split('\t')
                    term = convert_to_unicode(term)
                    self.term_dict[term] = int(term_id)

        self.inited_dict = True
        sys.stderr.write("loaded term dict:%s\n" % (len(self.term_dict)))

    def _pad_batch_data(self, insts, pad_idx, return_max_len=True, return_num_token=False):
        """
        Pad the instances to the max sequence length in batch, and generate the
        corresponding position data and attention bias.
        """
        return_list = []
        max_len = max(len(inst) for inst in insts)
        # Any token included in dict can be used to pad, since the paddings' loss
        # will be masked out by weights and make no effect on parameter gradients.
        inst_data = np.array(
            [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
        return_list += [inst_data.astype("int64").reshape([-1, 1])]
        
        if return_max_len:
            return_list += [max_len]
        if return_num_token:
            num_token = 0
            for inst in insts:
                num_token += len(inst)
            return_list += [num_token]
        return return_list if len(return_list) > 1 else return_list[0]

    def _get_ids(self, seg_info):
        if len(seg_info) < 1:
            return [0], [0]
        bt = seg_info.split('\3') 
        if len(self.term_dict) < 1:
            letter_ids = map(int, bt[0].split())[:self._flags.max_seq_len]
            word_ids = map(int, bt[1].split())[:self._flags.max_seq_len]
            return letter_ids, word_ids

        rq = convert_to_unicode("".join(bt))
        bl = [t for t in rq]
        letter_ids = [] 
        for t in bl:
            letter_ids.append(self.term_dict.get(t.lower(), 1))
            if len(letter_ids) >= self._flags.max_seq_len:
                break

        word_ids = []
        for t in bt:
            t = convert_to_unicode(t)
            word_ids.append(self.term_dict.get(t.lower(), 1)) 
            if len(word_ids) >= self._flags.max_seq_len:
                break
        return letter_ids, word_ids
 
    def _get_poi_ids(self, poi_str, max_num=0):
        if len(poi_str) < 1:
            return []
        ids = []
        all_p = poi_str.split('\1')
        
        pidx = range(0, len(all_p))
        if max_num > 0 and len(all_p) > max_num:
            #neg sample: last 10 is negative sampling(not disp)
            neg_s_idx = len(all_p) - 10
            pidx = [1, 2] + list(random.sample(pidx[3:neg_s_idx], max_num - 13)) + list(pidx[neg_s_idx:]) 
        bids = set()
        for x in pidx:
            poi_seg = all_p[x].split('\2')
            #raw_text: name, addr, xy
            bid = poi_seg[0]
            name_letter_id, name_word_id = self._get_ids(poi_seg[0])
            addr_letter_id, addr_word_id = self._get_ids(poi_seg[1])
            ghid = list(map(int, poi_seg[2].split(','))) 

            if not self.is_training and name_letter_id == [0]:
                continue # empty name
            if bid in bids:
                continue
            bids.add(bid)
            ids.append([name_letter_id, name_word_id, addr_letter_id, addr_word_id, ghid])

        return ids

    def deal_timestamp(self, timestamp):
        day_time_dt = datetime.datetime.fromtimestamp(timestamp)
        day = day_time_dt.strftime("%w")
        time = day_time_dt.strftime("%H.%M")
        day_id = int(day) * 2
        if 6 < float(time) < 18:
            return day_id
        else:
            return day_id + 1
    
    def parse_batch(self, data_gen):
        """
        reader_batch must be true: only for train & loss_func is log_exp, other use parse_oneline
        pos : neg = 1 : N
        """
        def _get_lod(k):
            return fluid.create_lod_tensor(np.array(batch_data[k][0]).reshape([-1, 1]),
                    [batch_data[k][1]], self.place)
        
        batch_data = {}
        keys = None
        last_gh = None
        task_data = None
        process_batch = False
        for gh, line in data_gen:
            # print(gh)
            # print(last_gh)
            if last_gh == None:
                last_gh = gh
                task_data = [line]
            elif last_gh != gh:
                last_gh = gh
                process_batch = True
            else:
                task_data.append(line)

            if process_batch:
                # print(len(task_data))
                gen_data = []
                for task_line in task_data:
                    # print(task_line)
                    for s in self.parse_oneline(task_line):
                        for k, v in s:
                            if k not in batch_data:
                                batch_data[k] = [[], []]
                            
                            if not isinstance(v[0], list):
                                v = [v] #pos 1 to N
                            for j in v:
                                batch_data[k][0].extend(j)
                                batch_data[k][1].append(len(j))

                        if keys is None:
                            keys = [k for k, _ in s]

                        if len(batch_data[keys[0]][1]) == self._flags.train_batch_size:
                            # print(keys)
                            gen_data.append([(k, _get_lod(k)) for k in keys])
                            batch_data = {}
                # if not self._flags.drop_last_batch and len(batch_data) != 0:
                #     gen_data.append([(k, _get_lod(k)) for k in keys])
                
                # print(gen_data)
                task_data = [line]
                process_batch = False
                if len(gen_data):
                    # print(len(gen_data))
                    yield gen_data
        
        # if not self._flags.drop_last_batch and len(batch_data) != 0:
        #     yield [(k, _get_lod(k)) for k in keys]

    def parse_oneline(self, line):
        """
        datareader interface
        """

        self._init_dict()
        qid, timestamp, gh, prefix, pos_poi, neg_poi = line.strip("\r\n").split("\t")
        # day_id = self.deal_timestamp(float(timestamp))
        # day_input = [0] * 14
        # day_input[day_id] = 1
        logid = int(qid.split('_')[1])
        #step2
        prefix_loc_geoid = list(map(int, gh.split(',')))
        prefix_letter_id, prefix_word_id = self._get_ids(prefix)
        prefix_input = [("prefix_letter_id", prefix_letter_id)]
        if self._flags.prefix_word_id:
            prefix_input.append(("prefix_word_id", prefix_word_id))
        if self._flags.use_geohash:
            prefix_input.append(("prefix_loc_geoid", prefix_loc_geoid))

        #step3
        pos_ids = self._get_poi_ids(pos_poi)
        pos_num = len(pos_ids)
        max_num = 0
        if self.is_training:
            max_num = max(20, self._flags.neg_sample_num) #last 10 is neg sample
        neg_ids = self._get_poi_ids(neg_poi, max_num=max_num)
        #if not train, add all pois
        if not self.is_training:
            pos_ids.extend(neg_ids[:-10] if len(neg_ids) > 10 else neg_ids)
            if len(pos_ids) < 1:
                pos_ids.append([[0], [0], [0], [0], [0] * 40, [0]])
        #step4
        idx = 0
        for pos_id in pos_ids:
            pos_input = [("pos_name_letter_id", pos_id[0]), \
                        ("pos_addr_letter_id", pos_id[2])]
            if self._flags.poi_word_id:
                pos_input.append(("pos_name_word_id", pos_id[1]))
                pos_input.append(("pos_addr_word_id", pos_id[3]))
            if self._flags.use_geohash:
                pos_input.append(("pos_loc_geoid", pos_id[4]))

            if self.is_training:
                if len(neg_ids) > self._flags.neg_sample_num:
                    #Noise Contrastive Estimation
                    #if self._flags.neg_sample_num > 3:
                    #    nids_sample = neg_ids[:3]
                    nids_sample = random.sample(neg_ids, self._flags.neg_sample_num)
                else:
                    nids_sample = neg_ids

                if self._flags.reader_batch:
                    if len(nids_sample) != self._flags.neg_sample_num:
                        continue

                    neg_batch = [[], [], [], [], []]
                    for neg_id in nids_sample:
                        for i in range(len(neg_batch)):
                            neg_batch[i].append(neg_id[i]) 
                    
                    neg_input = [("neg_name_letter_id", neg_batch[0]), \
                                ("neg_addr_letter_id", neg_batch[2])]
                    if self._flags.poi_word_id:
                        neg_input.append(("neg_name_word_id", neg_batch[1]))
                        neg_input.append(("neg_addr_word_id", neg_batch[3]))
                        
                    if self._flags.use_geohash:
                        neg_input.append(("neg_loc_geoid", neg_batch[4]))
                    yield prefix_input + pos_input + neg_input
                else:
                    for neg_id in nids_sample:
                        neg_input = [("neg_name_letter_id", neg_id[0]), \
                                    ("neg_addr_letter_id", neg_id[2])]
                        if self._flags.poi_word_id:
                            neg_input.append(("neg_name_word_id", neg_id[1]))
                            neg_input.append(("neg_addr_word_id", neg_id[3]))
                        if self._flags.use_geohash:
                            neg_input.append(("neg_loc_geoid", neg_id[4]))
                        yield prefix_input + pos_input + neg_input
            else:
                label = int(idx < pos_num)
                yield prefix_input + pos_input + [("label", [label]), ("qid", [logid])]

            idx += 1


# if __name__ == '__main__':
#     from utils import flags
#     from utils.load_conf_file import LoadConfFile
#     FLAGS = flags.FLAGS
#     flags.DEFINE_custom("conf_file", "./conf/test/test.conf", 
#         #"conf file", action=LoadConfFile, sec_name="Train")
#         "conf file", action=LoadConfFile, sec_name="Evaluate")
    
#     sys.stderr.write('-----------  Configuration Arguments -----------\n')
#     for arg, value in sorted(flags.get_flags_dict().items()):
#         sys.stderr.write('%s: %s\n' % (arg, value))
#     sys.stderr.write('------------------------------------------------\n')
   
#     dataset_instance = PoiQacPersonalized(FLAGS)
#     def _dump_vec(data, name):
#         print("%s\t%s" % (name, " ".join(map(str, np.array(data)))))
    
#     def _data_generator(): 
#         """
#         stdin sample generator: read from stdin 
#         """
#         for line in sys.stdin:
#             if not line.strip():
#                 continue
#             yield line

#     if FLAGS.reader_batch: 
#         for sample in dataset_instance.parse_batch(_data_generator):
#             _dump_vec(sample[0][1], 'prefix_letter_id')
#             _dump_vec(sample[1][1], 'prefix_loc_geoid')
#             _dump_vec(sample[2][1], 'pos_name_letter_id')
#             _dump_vec(sample[3][1], 'pos_addr_letter_id')
#             _dump_vec(sample[6][1], 'pos_loc_geoid')
#             _dump_vec(sample[7][1], 'neg_name_letter_id or label')
#     else:
#         for line in sys.stdin:
#             for sample in dataset_instance.parse_oneline(line):
#                 _dump_vec(sample[0][1], 'prefix_letter_id')
#                 _dump_vec(sample[1][1], 'prefix_loc_geoid')
#                 _dump_vec(sample[2][1], 'pos_name_letter_id')
#                 _dump_vec(sample[3][1], 'pos_addr_letter_id')
#                 _dump_vec(sample[6][1], 'pos_loc_geoid')
#                 _dump_vec(sample[7][1], 'neg_name_letter_id or label')

