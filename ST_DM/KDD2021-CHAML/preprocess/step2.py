import paddle
import time
import os
import pickle
import random
from constants import *
"""
step2: split the train/valid/test data for each base/valid/target city.
"""
if __name__ == '__main__':
    start = time.time()
    if not os.path.exists(split_path):
        os.mkdir(split_path)

    def get_user_num(mode):
        if mode == 'meta_training':
            return TRAIN_USER_NUM
        else:
            return TEST_USER_NUM

    def split_data(city, mode='meta_training'):

        def poi_type_to_id(poi_type):
            if poi_type == '' or poi_type == 'NONE':
                poi_type = '<PAD>'
            if poi_type in poitype_to_id:
                return poitype_to_id[poi_type]
            else:
                return 0
        data_path = dataset_path + '{}.txt'.format(city)
        pkl_path = root_path + 'pkls/{}/'.format(city)
        userid_to_id = pickle.load(open(pkl_path + 'userid_to_id.pkl', 'rb'))
        poiid_to_id = pickle.load(open(pkl_path + 'poiid_to_id.pkl', 'rb'))
        poitype_to_id = pickle.load(open(pkl_path + 'poitype_to_id.pkl', 'rb'))
        poiid_set = set(poiid_to_id.keys())
        poiid_set.remove('<PAD>')
        support_set_file = '{}_{}_support.txt'.format(city, mode)
        query_set_file = '{}_{}_query.txt'.format(city, mode)
        min_hist = 2 + MIN_HIST
        max_hist = 2 + MAX_HIST
        with open(split_path + support_set_file, 'w', encoding='utf-8'
            ) as support_out:
            with open(split_path + query_set_file, 'w', encoding='utf-8'
                ) as query_out:
                with open(data_path, 'r', encoding='utf-8') as origin_data:
                    user_num = get_user_num(mode)
                    i = 0
                    for line in origin_data.readlines():
                        if i >= user_num:
                            break
                        arr = line.strip().split('\t')
                        if len(arr) < max_hist:
                            continue
                        if len(arr) > max_hist:
                            arr = [arr[0], arr[1]] + arr[-(max_hist - 2):]
                        arr[0] = str(userid_to_id[arr[0]])
                        for k, search_record in enumerate(arr):
                            if k >= 2:
                                search_record = search_record.split('_')
                                search_record[1] = str(poiid_to_id[
                                    search_record[1]])
                                search_record[4] = str(poi_type_to_id(
                                    search_record[4]))
                                arr[k] = '_'.join(search_record)
                        support_out.write('\t'.join(arr[:min_hist +
                            SPT_SIZE]) + '\n')
                        query_out.write('\t'.join(arr) + '\n')
                        i += 1
    for city in get_cities('base'):
        split_data(city, 'meta_training')
    for city in get_cities('valid'):
        split_data(city, 'meta_testing')
    for city in get_cities('target'):
        split_data(city, 'meta_testing')
