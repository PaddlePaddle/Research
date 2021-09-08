import paddle
import pickle
import numpy as np
import os
from constants import *
import random
"""
step1: make the pickles (for matching data to integer numbers) of each city.
"""
if __name__ == '__main__':
    max_hist = MAX_HIST + 2
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    if not os.path.exists(root_path + 'pkls'):
        os.mkdir(root_path + 'pkls')
    poiid_to_id = {'<PAD>': 0}
    userid_to_id = {'<PAD>': 0}
    poitype_to_id = {'<PAD>': 0}
    poiid_to_name = {}
    poiid_to_loc = {}
    poiid_to_type = {}
    poi_type_file = config_path + 'poi_category.txt'
    with open(poi_type_file, 'r', encoding='utf-8') as f:
        for line in f:
            poi_type = line.strip()
            poitype_to_id[poi_type] = len(poitype_to_id)
    cities = []
    for filename in os.listdir(dataset_path):
        if '.txt' in filename:
            cities.append(filename.split('.')[0])

    def read_to_dict(data_path):
        with open(data_path, 'r', encoding='utf-8') as input:
            cnt = 0
            for line in input.readlines():
                arr = line.strip().split('\t')
                if len(arr) < max_hist:
                    continue
                if arr[0] not in userid_to_id:
                    userid_to_id[arr[0]] = len(userid_to_id)
                for search_record in arr[2:]:
                    search_record = search_record.split('_')
                    poiid = search_record[1]
                    poi_type = search_record[4]
                    if poi_type == 'NONE' or poi_type == '':
                        poi_type = '<PAD>'
                    poiid_to_loc[poiid] = search_record[2]
                    poiid_to_type[poiid] = poi_type
                    poiid_to_name[poiid] = search_record[3]
                    if poiid not in poiid_to_id:
                        poiid_to_id[poiid] = len(poiid_to_id)
                cnt += 1
                if cnt >= MAX_USER_TOTAL:
                    break
    for city in cities:
        data_path = dataset_path + '{}.txt'.format(city)
        pkl_path = root_path + 'pkls/{}/'.format(city)
        if not os.path.exists(pkl_path):
            os.mkdir(pkl_path)
        read_to_dict(data_path)
        pickle.dump(userid_to_id, open(pkl_path + 'userid_to_id.pkl', 'wb'))
        pickle.dump(poiid_to_id, open(pkl_path + 'poiid_to_id.pkl', 'wb'))
        pickle.dump(poitype_to_id, open(pkl_path + 'poitype_to_id.pkl', 'wb'))
        pickle.dump(poiid_to_loc, open(pkl_path + 'poiid_to_loc.pkl', 'wb'))
        pickle.dump(poiid_to_type, open(pkl_path + 'poiid_to_type.pkl', 'wb'))
        pickle.dump(poiid_to_name, open(pkl_path + 'poiid_to_name.pkl', 'wb'))
        poiid_to_id = {'<PAD>': 0}
        userid_to_id = {'<PAD>': 0}
        poiid_to_name = {}
        poiid_to_loc = {}
        poiid_to_type = {}
