import paddle
import pickle
import time
import numpy as np
import copy
import os
import random
from constants import *
"""
step3: make the final dataset files for transfer learning process and meta-learning.
For meta-learning, the output is 'mtrain_tasks.pkl' 'mvalid_tasks.pkl' 'mtest_tasks.pkl'
"""
start = time.time()


def timestr_to_timeid(timestr):
    if len(timestr) > 5:
        return time.strptime(timestr, '%Y-%m-%d %H:%M:%S').tm_hour + 1
    else:
        return 0


def timestr_to_timestamp(timestr):
    return int(time.mktime(time.strptime(timestr, '%Y-%m-%d %H:%M:%S')))


def cal_distance(coo1, coo2):
    x1, y1 = coo1[1:-1].split(',')
    x2, y2 = coo2[1:-1].split(',')
    dist = (float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2
    if dist > 10000000.0:
        dist = 10000000.0
    return dist


def cal_dtime(last_timestr, timestr):
    last_timestamp = timestr_to_timestamp(last_timestr)
    timestamp = timestr_to_timestamp(timestr)
    dtime = timestamp - last_timestamp
    return dtime


def get_mean_std(arr):
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)
    return mean, std


def record_to_datapoint(record):
    record = record.split('_')
    datapoint = [int(record[1]), int(record[4]), timestr_to_timeid(record[0])]
    return datapoint, record[0], record[2], record[-2]


def pad_one_seq(x, max_context):
    context_element_len = len(x[0])
    ret = [[0] * context_element_len] * (max_context - len(x)) + x
    return np.array(ret)[-max_context:]


def get_user_num(dataset):
    if 'meta_training' in dataset:
        return TRAIN_USER_NUM
    else:
        return TEST_USER_NUM


def read_dataset(dataset, min_hist=MIN_HIST):
    task_user2samples = {}
    task_candidates = set()
    user2itemset = {}
    dists = []
    dtimes = []
    user_num = get_user_num(dataset)
    with open(split_path + dataset, 'r', encoding='utf-8') as fin:
        for cnt, line1 in enumerate(fin):
            if cnt >= user_num:
                break
            arr1 = line1.strip().split('\t')
            uid = int(arr1.pop(0))
            arr1.pop(0)
            user2itemset[uid] = set()
            context = []
            user_loc_context = []
            last_timestr = 'NULL'
            for i in range(0, len(arr1)):
                datapoint, timestr, poi_loc, user_loc = record_to_datapoint(
                    arr1[i])
                dist = cal_distance(poi_loc, user_loc)
                dists.append(dist)
                if i == 0:
                    context.append(datapoint + [dist, 0])
                else:
                    dtime = cal_dtime(last_timestr, timestr)
                    dtimes.append(dtime)
                    context.append(datapoint + [dist, dtime])
                candi_tuple = tuple([datapoint[0], datapoint[1], poi_loc])
                task_candidates.add(candi_tuple)
                user2itemset[uid].add(candi_tuple)
                last_timestr = timestr
                user_loc_context.append(user_loc)
            for j in range(0, len(arr1) - min_hist):
                hist = np.array(pad_one_seq(context[0:min_hist + j],
                    MAX_HIST), dtype='int32')
                candi = np.array(context[min_hist + j], dtype='int32')
                sample = np.array([uid, hist, candi, user_loc_context[
                    min_hist + j], 1])
                if uid in task_user2samples:
                    task_user2samples[uid].append(sample)
                else:
                    task_user2samples[uid] = [sample]
    task_candidates = list(task_candidates)
    mean_stds = {}
    mean_stds['dist'] = mean_dist, std_dist = get_mean_std(dists)
    mean_stds['dtime'] = mean_dtime, std_dtime = get_mean_std(dtimes)
    data = task_user2samples, task_candidates, user2itemset, mean_stds
    return data


def read_transfer_data(datasets, neg_num=1, min_hist=MIN_HIST, is_test_qry=\
    False):
    for i in range(len(datasets)):
        city = datasets[i].split('_')[0]
        task_user2pos_samples, task_candidates, user2itemset, mean_stds = (
            read_dataset(datasets[i], min_hist))
        user_list = []
        history_list = []
        candidate_list = []
        label_list = []
        test_qry_samples = []
        for user in task_user2pos_samples:
            for pos_sample in task_user2pos_samples[user]:
                user_id, hist, pos_candi, user_loc, label = pos_sample
                for k in range(1 + neg_num):
                    user_list.append(user_id)
                    history_list.append(hist)
                candidate_list.append(pos_candi)
                label_list.append(1)
                neg_candis = []
                neg_qry_samples = []
                for k in range(neg_num):
                    neg_candi = random.choice(task_candidates)
                    while neg_candi in user2itemset[user_id
                        ] or neg_candi in neg_candis:
                        neg_candi = random.choice(task_candidates)
                    neg_candis.append(neg_candi)
                    poiid, poitype, poi_loc = neg_candi
                    neg_candi = np.array([poiid, poitype, pos_candi[2],
                        cal_distance(poi_loc, user_loc), pos_candi[4]])
                    candidate_list.append(neg_candi)
                    label_list.append(0)
                    if is_test_qry:
                        neg_qry_samples.append(np.array([user_id, hist,
                            neg_candi, 0]))
                if is_test_qry:
                    pos_sample = user_id, hist, pos_candi, label
                    test_qry_samples.extend([pos_sample] + neg_qry_samples)
        if is_test_qry:
            pickle.dump(test_qry_samples, open(split_path + city +
                '_test_qry_samples.pkl', 'wb'), protocol=4)
        pickle.dump(mean_stds, open(save_path + city + '_mean_stds.pkl',
            'wb'), protocol=4)
        yield np.array(user_list, dtype='int32'), np.array(history_list,
            dtype='int32'), np.array(candidate_list, dtype='int32'), np.array(
            label_list, dtype='int32')


def save_train_and_valid(data, city, mode, valid_ratio=0.05):
    x_train_uid, x_train_history, x_train_candi, y_train = data
    train_filename = '{}_{}_train.pkl'.format(city, mode)
    valid_filename = '{}_{}_dev.pkl'.format(city, mode)
    TRAIN_SAMPLE_NUM = int(len(x_train_uid) * (1 - valid_ratio))
    pickle.dump([x_train_uid[:TRAIN_SAMPLE_NUM], x_train_history[:
        TRAIN_SAMPLE_NUM], x_train_candi[:TRAIN_SAMPLE_NUM], y_train[:
        TRAIN_SAMPLE_NUM]], open(save_path + train_filename, 'wb'), protocol=4)
    pickle.dump([x_train_uid[TRAIN_SAMPLE_NUM:], x_train_history[
        TRAIN_SAMPLE_NUM:], x_train_candi[TRAIN_SAMPLE_NUM:], y_train[
        TRAIN_SAMPLE_NUM:]], open(save_path + valid_filename, 'wb'), protocol=4
        )


def save_test(data, city):
    x_test_uid, x_test_history, x_test_candi, y_test = data
    test_filename = '{}_target_test.pkl'.format(city)
    pickle.dump([x_test_uid, x_test_history, x_test_candi, y_test], open(
        save_path + test_filename, 'wb'), protocol=4)


def generate_base_cities(cities):
    datasets = list(map(lambda x: x + '_meta_training_query.txt', cities))
    for i, data in enumerate(read_transfer_data(datasets, neg_num=1)):
        save_train_and_valid(data, cities[i], 'base')


def generate_valid_cities_as_train(cities):
    spt_datasets = list(map(lambda x: x + '_meta_testing_support.txt', cities))
    qry_datasets = list(map(lambda x: x + '_meta_testing_query.txt', cities))
    data_collect = [[] for i in range(len(cities))]
    for i, data in enumerate(read_transfer_data(spt_datasets, neg_num=1)):
        data_collect[i].append(data)
    for i, data in enumerate(read_transfer_data(qry_datasets, neg_num=1,
        min_hist=MIN_HIST + SPT_SIZE)):
        data_collect[i].append(data)
    for i in range(len(cities)):
        spt_data, qry_data = data_collect[i]
        data = []
        for j in range(4):
            data.append(np.concatenate(np.array([spt_data[j], qry_data[j]]),
                axis=0))
        save_train_and_valid(data, cities[i], 'base')


def generate_target_cities(cities):
    spt_datasets = list(map(lambda x: x + '_meta_testing_support.txt', cities))
    qry_datasets = list(map(lambda x: x + '_meta_testing_query.txt', cities))
    for i, data in enumerate(read_transfer_data(spt_datasets, neg_num=1)):
        save_train_and_valid(data, cities[i], 'target')
    for i, data in enumerate(read_transfer_data(qry_datasets, neg_num=100,
        min_hist=MIN_HIST + SPT_SIZE, is_test_qry=True)):
        save_test(data, cities[i])


def read_meta_training_data(cities):
    spt_datasets = list(map(lambda x: x + '_meta_training_support.txt', cities)
        )
    qry_datasets = list(map(lambda x: x + '_meta_training_query.txt', cities))
    mtrain_tasks = []
    for i in range(len(cities)):
        (spt_user2samples, spt_candidates, spt_user2itemset, spt_mean_stds
            ) = read_dataset(spt_datasets[i])
        (qry_user2samples, qry_candidates, qry_user2itemset, qry_mean_stds
            ) = read_dataset(qry_datasets[i], min_hist=MIN_HIST + SPT_SIZE)
        task_candidates = list(set(spt_candidates) | set(qry_candidates))
        mtrain_tasks.append((spt_user2samples, qry_user2samples,
            task_candidates, qry_user2itemset, qry_mean_stds, cities[i]))
    return mtrain_tasks


def read_meta_testing_data(cities, is_test=False):
    spt_datasets = list(map(lambda x: x + '_meta_testing_support.txt', cities))
    qry_datasets = list(map(lambda x: x + '_meta_testing_query.txt', cities))
    mtest_tasks = []
    for i in range(len(spt_datasets)):
        (spt_user2samples, spt_candidates, spt_user2itemset, spt_mean_stds
            ) = read_dataset(spt_datasets[i])
        (qry_user2samples, qry_candidates, qry_user2itemset, qry_mean_stds
            ) = read_dataset(qry_datasets[i], min_hist=MIN_HIST + SPT_SIZE)
        task_candidates = list(set(spt_candidates) | set(qry_candidates))
        if is_test:
            city = cities[i]
            align_qry_samples = pickle.load(open(split_path + city +
                '_test_qry_samples.pkl', 'rb'))
            mtest_tasks.append((spt_user2samples, qry_user2samples,
                task_candidates, qry_user2itemset, qry_mean_stds, cities[i],
                align_qry_samples))
        else:
            mtest_tasks.append((spt_user2samples, qry_user2samples,
                task_candidates, qry_user2itemset, qry_mean_stds, cities[i]))
    return mtest_tasks


def generate_trans():
    generate_base_cities(base_cities)
    generate_target_cities(target_cities)
    generate_valid_cities_as_train(valid_cities)


def generate_meta():
    pickle.dump(read_meta_training_data(base_cities), open(save_path +
        'mtrain_tasks.pkl', 'wb'), protocol=4)
    pickle.dump(read_meta_testing_data(valid_cities), open(save_path +
        'mvalid_tasks.pkl', 'wb'), protocol=4)
    pickle.dump(read_meta_testing_data(target_cities, is_test=True), open(
        save_path + 'mtest_tasks.pkl', 'wb'), protocol=4)


if not os.path.exists(save_path):
    os.mkdir(save_path)
base_cities = get_cities('base')
valid_cities = get_cities('valid')
target_cities = get_cities('target')
print('generate the final dataset for transfer methods.')
generate_trans()
print('generate the final dataset for meta-learning methods.')
generate_meta()
end = time.time()
print('cost time:', (end - start) / 60, 'min')
