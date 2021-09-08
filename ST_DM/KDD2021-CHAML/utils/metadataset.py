import numpy as np
import os
import pickle
import paddle
import copy
import random
import math
SPT_SIZE = 2


def samples_to_input(samples):
    x_uid = []
    x_hist = []
    x_candi = []
    y = []
    for i in range(len(samples)):
        x_uid.append(samples[i][0])
        x_hist.append(samples[i][1])
        x_candi.append(samples[i][2])
        y.append(samples[i][3])
    x_uid = paddle.to_tensor(np.array(x_uid), dtype='int64')
    x_hist = paddle.to_tensor(np.array(x_hist), dtype='int64')
    x_candi = paddle.to_tensor(np.array(x_candi), dtype='int64')
    y = paddle.to_tensor(np.array(y), dtype='int64')
    x = [x_uid, x_hist, x_candi]
    return x, y


def cal_distance(coo1, coo2):
    x1, y1 = coo1[1:-1].split(',')
    x2, y2 = coo2[1:-1].split(',')
    dist = (float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2
    if dist > 10000000.0:
        dist = 10000000.0
    return dist


def cal_distance_latlon(lat1, lon1, lat2, lon2):
    dist = math.sqrt((float(lat1) - float(lat2)) ** 2 + (float(lon1) -
        float(lon2)) ** 2)
    return dist


def pos_samples2pos_and_neg(pos_spt_samples, task_candidates,
    task_user2itemset, negative_ratio=1):
    final_pos_samples = []
    final_neg_samples = []
    for i in range(len(pos_spt_samples)):
        pos_sample = pos_spt_samples[i]
        user_id, hist, pos_candi, user_loc, label = pos_sample
        neg_candi = random.choice(task_candidates)
        while neg_candi in task_user2itemset[user_id]:
            neg_candi = random.choice(task_candidates)
        poiid, poitype, poi_loc = neg_candi
        neg_candi = np.array([poiid, poitype, pos_candi[2], cal_distance(
            poi_loc, user_loc), pos_candi[4]])
        final_pos_samples.append(np.array([user_id, hist, pos_candi, label]))
        final_neg_samples.append(np.array([user_id, hist, neg_candi, 0]))
    return final_pos_samples + final_neg_samples


def append_one_task(x_spt, y_spt, x_qry, y_qry, task_poiid_emb):
    global x_uid_spts, x_hist_spts, x_candi_spts, y_spts, x_uid_qrys, x_hist_qrys, x_candi_qrys, y_qrys, poiid_embs
    x_uid_spts.append(x_spt[0])
    x_hist_spts.append(x_spt[1])
    x_candi_spts.append(x_spt[2])
    y_spts.append(y_spt)
    x_uid_qrys.append(x_qry[0])
    x_hist_qrys.append(x_qry[1])
    x_candi_qrys.append(x_qry[2])
    y_qrys.append(y_qry)
    poiid_embs.append(task_poiid_emb)


def init_yield_collectors():
    global x_uid_spts, x_hist_spts, x_candi_spts, y_spts, x_uid_qrys, x_hist_qrys, x_candi_qrys, y_qrys, poiid_embs
    x_uid_spts, x_hist_spts, x_candi_spts, y_spts = [], [], [], []
    x_uid_qrys, x_hist_qrys, x_candi_qrys, y_qrys = [], [], [], []
    poiid_embs = []


def get_task_poiid_embs(id_emb_path, cities):
    task_poiid_embs = []
    for idx in range(len(cities)):
        poiid_emb_file = '{}{}_poiid_embed.npy'.format(id_emb_path, cities[idx]
            )
        # poiid_emb = torch.from_numpy(np.load(poiid_emb_file).astype(np.float32)
        #     )
        poiid_emb = paddle.to_tensor(np.load(poiid_emb_file).astype(np.float32))
        task_poiid_embs.append(poiid_emb)
    return task_poiid_embs


def task_to_cities(tasks, index=-1):
    cities = []
    for idx in range(len(tasks)):
        cities.append(tasks[idx][index])
    return cities


class TrainGenerator:

    def __init__(self, root_path, meta_path, id_emb_path, mtrain_tasks,
        batch_size, task_batch_size, curriculum_task_idxs, pacing_function=\
        'ssp', few_num=512, max_steps=None, negative_ratio=1):
        self.cities = task_to_cities(mtrain_tasks, index=-1)
        self.task_num = len(self.cities)
        self.task_poiid_embs = get_task_poiid_embs(id_emb_path, self.cities)
        self.few_user_num = few_num // SPT_SIZE
        self.negative_ratio = negative_ratio
        self.task_batch_size = task_batch_size
        self.mtrain_tasks = mtrain_tasks
        self.curriculum_task_idxs = curriculum_task_idxs
        self.max_steps = max_steps
        self.pacing_function = pacing_function

    def fetch_task_batch(self, task_idx2results=None, stage='stage1',
        curriculum=True, hard_task=True, batch_id=0):
        task_idx2acc = task_idx2results['task_idx2acc']
        task_idx_to_user2acc = task_idx2results['task_idx_to_user2acc']

        # 1. City-level curriculum decides the sampling pool of meta-training tasks.
        if curriculum:
            if self.pacing_function == 'ssp':
                starting_percent = 0.5
                step_length = self.max_steps // 2  # max_"steps": max iteration "steps"
                if batch_id < step_length:
                    gi = int(starting_percent * self.task_num)
                else:
                    gi = self.task_num
                task_idx_pool = self.curriculum_task_idxs[:gi]
        else:
            task_idx_pool = list(range(self.task_num))
        self.task_idx_pool = task_idx_pool
        self.last_batch_id = batch_id

        # 2. Decide to sample which cities/tasks for this round.
        if task_idx2acc is None:  # None: means it's the first iteration of meta-training, then we randomly sample tasks from the current city pool
            task_idxs = random.sample(task_idx_pool, k=self.task_batch_size)
        elif stage == 'stage2':  # For Stage2, we keep the same group of cities (tasks), and sample new users
            task_idxs = list(task_idx2acc.keys())
        elif hard_task:  # [Flag][hard_task]: for the rest of meta-training iterations, if we adopt hard_task strategy, we try to sample harder cities
            hard_task_num = self.task_batch_size // 2  # $k_c$ is 0.5, $B_c$ is self.task_batch_size.
            task_idxs = list(task_idx2acc.keys())[:hard_task_num]
            if len(task_idx_pool
                ) - self.task_batch_size < self.task_batch_size - hard_task_num:
                task_idxs = random.sample(task_idx_pool, k=self.task_batch_size
                    )
            else:
                other_task_pool = list(set(task_idx_pool) - set(list(
                    task_idx2acc.keys())))
                other_task_idxs = random.sample(other_task_pool, k=self.
                    task_batch_size - hard_task_num)
                task_idxs.extend(other_task_idxs)
        else:
            task_idxs = random.sample(task_idx_pool, k=self.task_batch_size)

        # 3. Decide the users of each city (to form support&query set samples).
        task_sample_sub2user = []  # each task -> a list x, x[i] means the user id of the i-th sample
        task_cont_feat_scalers = []  # each task -> a dict: {"dist": (mean, std), "dtime": (mean, std)}
        global x_uid_spts, x_hist_spts, x_candi_spts, y_spts, x_uid_qrys, x_hist_qrys, x_candi_qrys, y_qrys, poiid_embs
        init_yield_collectors()
        for idx in task_idxs:
            (spt_user2samples, qry_user2samples, candidates, user2itemset,
                qry_mean_stds, city_name) = self.mtrain_tasks[idx]
            task_cont_feat_scalers.append(qry_mean_stds)
            all_users = list(spt_user2samples.keys())
            # [Flag][hard_user] if we adopt hard_user strategy 
            if (stage == 'stage2' and task_idx_to_user2acc is not None and 
                idx in task_idx_to_user2acc):
                user2acc = task_idx_to_user2acc[idx]
                hard_user_num = self.few_user_num // 2  # $k_u$ is 0.5, $B_u$ is self.few_user_num.
                selected_users = list(user2acc.keys())[:hard_user_num]
                other_user_pool = list(set(all_users) - set(list(user2acc.
                    keys())))
                other_users = random.sample(other_user_pool, k=self.
                    few_user_num - hard_user_num)
                selected_users.extend(other_users)
            else:
                selected_users = random.sample(all_users, k=self.few_user_num)
            # construct the support set samples of a meta-training task
            pos_spt_samples = []
            for user in selected_users:
                pos_spt_samples.extend(spt_user2samples[user])
            pos_and_neg_spt_samples = pos_samples2pos_and_neg(pos_spt_samples,
                candidates, user2itemset, self.negative_ratio)
            x_spt, y_spt = samples_to_input(pos_and_neg_spt_samples)
            # construct the query set samples of a meta-training task
            qrysample_sub2user = []
            pos_qry_samples = []
            for user in selected_users:
                qry_samples = qry_user2samples[user]
                pos_qry_samples.extend(qry_samples)
                qrysample_sub2user.extend([user] * len(qry_samples))
            qrysample_sub2user = qrysample_sub2user + qrysample_sub2user  # all samples = [postive samples] + [negative samples], two lists have the same order of users.
            task_sample_sub2user.append(qrysample_sub2user)
            pos_and_neg_qry_samples = pos_samples2pos_and_neg(pos_qry_samples,
                candidates, user2itemset, self.negative_ratio)
            x_qry, y_qry = samples_to_input(pos_and_neg_qry_samples)
            append_one_task(x_spt, y_spt, x_qry, y_qry, self.
                task_poiid_embs[idx])  # support, query samples: each have 128 * 2 samples
        return ([x_uid_spts, x_hist_spts, x_candi_spts], y_spts, [
            x_uid_qrys, x_hist_qrys, x_candi_qrys], y_qrys, poiid_embs
            ), task_idxs, task_sample_sub2user, task_cont_feat_scalers


def evaluate_generator(root_path, id_emb_path, mtest_tasks, few_num,
    neg_num=100, is_test=False):

    # mtest_tasks: a list，each element is the data of a meta-testing task (0 spt_samples, 1 qry_samples, 2 candidates, 3 spt_user2itemset, 4 qry_user2itemset, 5 city_name)

    def task_iterator(qry_user2samples, candidates, user2itemset,
        yield_batch_size, align_qry_samples=None):
        test_batch_samples = []
        if align_qry_samples is None:
            qry_samples = []
            for user in qry_user2samples:
                qry_samples.extend(qry_user2samples[user])
            for pos_qry_sample in qry_samples:
                user_id, hist, pos_candi, user_loc, label = pos_qry_sample
                neg_qry_samples = []
                neg_candis = []
                for k in range(neg_num):
                    neg_candi = random.choice(candidates)
                    while neg_candi in user2itemset[user_id
                        ] or neg_candi in neg_candis:
                        neg_candi = random.choice(candidates)
                    neg_candis.append(neg_candi)  # neg_candi is tuple (poiid, poitype)
                    poiid, poitype, poi_loc = neg_candi
                    # candidate (poiid, poitype, timeid, u-pdist, dtime)
                    neg_candi = np.array([poiid, poitype, pos_candi[2],
                        cal_distance(poi_loc, user_loc), pos_candi[4]])
                    neg_qry_samples.append(np.array([user_id, hist,
                        neg_candi, 0]))
                final_pos_qry_sample = np.array([user_id, hist, pos_candi,
                    label])
                test_batch_samples.extend([final_pos_qry_sample] +
                    neg_qry_samples)
                if len(test_batch_samples) >= yield_batch_size:
                    x_qry, y_qry = samples_to_input(test_batch_samples)
                    yield x_qry, y_qry
                    test_batch_samples = []
        else:
            for i in range(len(align_qry_samples) // yield_batch_size):
                test_batch_samples = align_qry_samples[i * yield_batch_size
                    :(i + 1) * yield_batch_size]
                x_qry, y_qry = samples_to_input(test_batch_samples)
                yield x_qry, y_qry
    if is_test:
        cities = task_to_cities(mtest_tasks, index=-2)
    else:
        cities = task_to_cities(mtest_tasks, index=-1)
    task_poiid_embs = get_task_poiid_embs(id_emb_path, cities)
    for idx in range(len(mtest_tasks)):
        if is_test:
            (spt_user2samples, qry_user2samples, task_candidates,
                qry_user2itemset, qry_mean_stds, city_name, align_qry_samples
                ) = mtest_tasks[idx]
        else:
            (spt_user2samples, qry_user2samples, task_candidates,
                qry_user2itemset, qry_mean_stds, city_name) = mtest_tasks[idx]
            align_qry_samples = None
        all_users = list(spt_user2samples.keys())
        if few_num is None:
            few_user_num = len(all_users)
            selected_users = all_users
        else:
            few_user_num = few_num // SPT_SIZE
            selected_users = random.sample(all_users, k=few_user_num)
        pos_spt_samples = []
        for user in selected_users:
            pos_spt_samples.extend(spt_user2samples[user])
        pos_and_neg_spt_samples = pos_samples2pos_and_neg(pos_spt_samples,
            task_candidates, qry_user2itemset, negative_ratio=1)
        x_spt, y_spt = samples_to_input(pos_and_neg_spt_samples)
        yield x_spt, y_spt, task_iterator(qry_user2samples, task_candidates,
            qry_user2itemset, yield_batch_size=101 * 20, align_qry_samples=\
            align_qry_samples), task_poiid_embs[idx], city_name, qry_mean_stds
