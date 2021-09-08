# from x2paddle import torch2paddle
from copy import deepcopy
import argparse
import math
import os
import pickle
import json
import logging
import time
import paddle
import numpy as np
import random
from collections import Counter
from paddle.nn import functional as F
from model.meta import Meta
from utils.metrics import Metrics
from utils.metadataset import TrainGenerator
from utils.metadataset import evaluate_generator
logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
    datefmt='%m/%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
ARG = argparse.ArgumentParser()
parser = ARG
ARG = parser.parse_args()
root_path = './data/'
meta_path = root_path + 'dataset/final/'


def filter_statedict(module):
    state_dict = module.state_dict(keep_vars=True)
    non_params = []
    for key, value in state_dict.items():
        if not value.requires_grad:
            non_params.append(key)
    state_dict = module.state_dict()
    for key in non_params:
        del state_dict[key]
    return state_dict


def get_cities(root_path, which='base'):
    config_path = root_path + 'config/'
    cities_file = config_path + which + '_cities.txt'
    cities = []
    with open(cities_file, 'r') as f:
        for line in f:
            city = line.strip()
            cities.append(city)
    return cities


def get_curriculum(root_path):
    """
    TODO: One has to instantiate the curriculum according to the paper.
    That means to 
        1. train the learner (DIN) model on the "transfer-learning training set" of each city independently; 
        2. save the validation scores to the log;
        3. rank the cities from easiest to hardest according to the scores and save the ranked city indices 
            as a numpy array to "base_task_hardness.pkl". E.g., [4, 2, 1, 0, 5, 3, 7, 6].
    """
    # config_path = root_path + "config/"
    # config_file = config_path + "base_task_hardness.pkl"
    # curriculum = pickle.load(open(config_file, 'rb'))
    # ret = list(curriculum)
    # return ret
    mtrain_city_num = len(get_cities(root_path, 'base'))
    return list(np.arange(mtrain_city_num))


def get_config(json_file, pkl_path=None):
    with open(json_file, 'r') as f:
        config = json.load(f)
    config['poiid_dim'] = POIID_DIM
    config['with_cont_feat'] = WITH_CONT_FEAT
    poitype_to_id = pickle.load(open(poi_type2idx_path, 'rb'))
    config['num_poi_types'] = len(poitype_to_id)
    if pkl_path is not None:
        userid_to_id = pickle.load(open(pkl_path + 'userid_to_id.pkl', 'rb'))
        poiid_to_id = pickle.load(open(pkl_path + 'poiid_to_id.pkl', 'rb'))
        config['num_users'] = len(userid_to_id)
        config['num_pois'] = len(poiid_to_id)
    logger.info('Got config from {}'.format(json_file))
    return config


def get_model(meta_path, config, model_name='Mymodel', load_model_file=None,
    poiid_emb_file=None):
    # update some paths to config
    model_save_path = meta_path + 'model_save/'
    loss_save_path = meta_path + 'loss_save/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    if not os.path.exists(loss_save_path):
        os.mkdir(loss_save_path)
    config['save_path'] = '{}.pdiparams'.format(model_save_path + model_name)
    config['loss_save_path'] = '{}loss_{}.txt'.format(loss_save_path,
        model_name)
    if not SCRATCH_ID_EMB:
        config['poiid_emb_file'] = poiid_emb_file
    # create model
    if model_name == 'Meta':
        model = None
        meta_model = Meta(config)
        return meta_model, model


def get_optimizer(meta_model, config):
    init_parameters = list(filter(lambda p: p.trainable, meta_model.net
        .parameters()))
    parameters = init_parameters
    clip = paddle.nn.ClipGradByValue(min=-0.25, max=0.25)
    optimizer = paddle.optimizer.Adam(parameters=parameters, learning_rate=\
        config['meta_lr'], grad_clip=clip)
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=config[
        'meta_lr'], mode='max', factor=0.2, patience=PATIENCE, verbose=True,
        min_lr=1e-06)
    return optimizer, scheduler


def task_to_device(x_spt, y_spt, x_qry, y_qry, poiid_embs, device):
    if type(y_spt) == list:
        for i in range(len(x_spt)):
            for j in range(len(x_spt[0])):
                x_spt[i][j] = x_spt[i][j].to(device)
                x_qry[i][j] = x_qry[i][j].to(device)
        for i in range(len(y_spt)):
            y_spt[i] = y_spt[i].to(device)
            y_qry[i] = y_qry[i].to(device)
            poiid_embs[i] = poiid_embs[i].to(device)
        return x_spt, y_spt, x_qry, y_qry, poiid_embs
    else:
        for i in range(len(x_spt)):
            x_spt[i] = x_spt[i].to(device)
            x_qry[i] = x_qry[i].to(device)
        return x_spt, y_spt.to(device), x_qry, y_qry.to(device), poiid_embs.to(
            device)


def evaluate(data_loader, meta_model, metric, device, model=None,
    init_compare=False, silence=False):
    task_scores = []
    task_score_weights = []  # the weight of each task in the final evaluation scores (according to the amount of query users)
    if not silence:
        print('\t'.join(['Hits@5', 'Hits@10', 'NDCG@5', 'NDCG@10', 'city']))
    for data in data_loader:  # evaluate on one meta-test city (task) each time
        x_spt, y_spt, task_iterator, poiid_emb, city_name, scaler = data  # one meta-test task
        for i in range(len(x_spt)):
            x_spt[i] = x_spt[i].to(device)
        y_spt = y_spt.to(device)
        poiid_emb = poiid_emb.to(device)
        init_weights = list(meta_model.net.parameters())
        init_weights[0] = poiid_emb
        fast_weights = meta_model.finetuning_adapt(x_spt, y_spt, poiid_emb,
            scaler=scaler)
        all_y_pred_prob = []
        all_y_truth = []
        init_all_y_pred_prob = []
        for x_qry, y_qry in task_iterator:
            for i in range(len(x_qry)):
                x_qry[i] = x_qry[i].to(device)
            y_qry = y_qry.to(device)
            y_pred, y_pred_prob = meta_model.finetuning_predict(x_qry,
                y_qry, fast_weights, poiid_emb, scaler=scaler)
            all_y_pred_prob.extend(y_pred_prob)
            all_y_truth.extend(y_qry.data.detach().cpu().numpy().tolist())
            if init_compare:
                y_pred, y_pred_prob = meta_model.finetuning_predict(x_qry,
                    y_qry, init_weights, poiid_emb, scaler=scaler)
                init_all_y_pred_prob.extend(y_pred_prob)
        scores = metric.compute_metric(all_y_pred_prob, all_y_truth,
            session_len=101)
        if not silence:
            if init_compare:
                init_scores = metric.compute_metric(init_all_y_pred_prob,
                    all_y_truth, session_len=101)
                print('(init) {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(*
                    init_scores, city_name))
                print('(adapt) {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(*
                    scores, city_name))
            else:
                print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(*scores,
                    city_name))
        task_scores.append(list(scores))
        task_score_weights.append(len(all_y_truth) / 101)
    return np.array(task_scores), np.array(task_score_weights)


def update_hardness(task_idxs, task_sample_sub2user, results):
    task_level_acc = results['task_level_acc']
    task_sample_level_corrects = results['task_sample_level_corrects']
    task_idx2acc = {}
    task_idx_to_user2acc = {}
    for i in range(len(task_idxs)):
        task_idx2acc[task_idxs[i]] = task_level_acc[i]
        sample_corrects = task_sample_level_corrects[i]
        sample_sub2user = task_sample_sub2user[i]
        user2acc = {}
        for j in range(len(sample_corrects)):
            user = sample_sub2user[j]
            if user in user2acc:
                user2acc[user].append(sample_corrects[j])
            else:
                user2acc[user] = [sample_corrects[j]]
        for user in user2acc:
            user_level_acc = sum(user2acc[user]) / len(user2acc[user])
            user2acc[user] = user_level_acc
        user2acc = dict(sorted(user2acc.items(), key=lambda x: x[1]))
        task_idx_to_user2acc[task_idxs[i]] = user2acc
    task_idx2acc = dict(sorted(task_idx2acc.items(), key=lambda x: x[1]))
    task_idx2results = {'task_idx2acc': task_idx2acc,
        'task_idx_to_user2acc': task_idx_to_user2acc}
    return task_idx2results


def one_meta_training_step(task_gen, meta_model, optimizer, device,
    parameters, task_idx2results, stage, curriculum, hard_task, batch_id):
    data, task_idxs, task_sample_sub2user, cont_feat_scalers = (task_gen.
        fetch_task_batch(task_idx2results=task_idx2results, stage=stage,
        curriculum=curriculum, hard_task=hard_task, batch_id=batch_id))
    x_spt, y_spt, x_qry, y_qry, poiid_embs = task_to_device(*data, device)
    accs, loss_q, results = meta_model(x_spt, y_spt, x_qry, y_qry,
        poiid_embs=poiid_embs, cont_feat_scalers=cont_feat_scalers)
    optimizer.clear_grad()
    loss_q.backward(retain_graph=True)
    # torch2paddle.clip_grad_value_(parameters, 0.25)
    optimizer.step()
    task_idx2results = update_hardness(task_idxs, task_sample_sub2user, results
        )
    # accs,loss_q.item() are for logging during training, task_idx2results supports the next meta training step
    return accs, loss_q.item(), task_idx2results


def main_meta(meta_path, root_path, id_emb_path, model_name='Meta'):
    # read config file
    config_path = 'config/config-'
    model2config = {'Meta': '{}{}.json'.format(config_path, 'chaml')}
    config = get_config(model2config[model_name])
    print(config)

    # get meta_model, optimizer, metrics
    meta_model, model = get_model(meta_path, config, model_name)
    optimizer, scheduler = get_optimizer(meta_model, config)
    device = 'cuda'
    device = device.replace('cuda', 'cpu')   # TODO: you may delete this row to allow GPU
    device = paddle.set_device(device)
    meta_model = meta_model.to(device)
    if model is not None:
        model.to(device)
    parameters = list(filter(lambda p: p.requires_grad, meta_model.
        parameters()))
    tmp = filter(lambda x: x.requires_grad, meta_model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(meta_model)
    logger.info('Total trainable tensors: {}'.format(num))
    metric = Metrics()

    # load the dataset
    # a list, each element of the list is the data of a meta-training task (samples, candidates, user2items)
    mtrain_tasks = pickle.load(open(meta_path + 'mtrain_tasks.pkl', 'rb')) 
    # a list, each element of the list is the data of a meta-valid task(spt_samples, qry_samples, candidates, qry_user2items)
    mvalid_tasks = pickle.load(open(meta_path + 'mvalid_tasks.pkl', 'rb'))
    mtest_tasks = pickle.load(open(meta_path + 'mtest_tasks.pkl', 'rb'))
    logger.info('Loaded all the data pickles!')

    # set variables for statistics
    best_scores = 0
    running_loss = 0
    batch_id = 0

    # start training
    running_accs = np.zeros(config['update_step'] + 1)
    task_gen = TrainGenerator(root_path, meta_path, id_emb_path,
        mtrain_tasks, config['train_qry_batch_size'], config[
        'task_batch_size'], curriculum_task_idxs=get_curriculum(root_path),
        pacing_function=PACING_FUNCTION, few_num=config['few_num'],
        max_steps=config['max_train_steps'])
    task_idx2results = {'task_idx2acc': None, 'task_idx_to_user2acc': None}
    hard_task_counter = Counter()
    while True:
        # >>>>> [Stage1] sample the hardest tasks of last round, and then sample more tasks. 
        accs, loss, task_idx2results = one_meta_training_step(task_gen,
            meta_model, optimizer, device, parameters, task_idx2results,
            'stage1', CURRICULUM, HARD_TASK, batch_id=batch_id)
        running_loss += loss
        running_accs += accs
        batch_id += 1
        hard_task_counter.update(list(task_idx2results['task_idx2acc'].keys()))
        if HARD_USER:
            # >>>>> Stage2: the same tasks as Stage 1, keep the hardest users, and sample new users in these tasks.
            accs, loss, task_idx2results = one_meta_training_step(task_gen,
                meta_model, optimizer, device, parameters, task_idx2results,
                'stage2', CURRICULUM, HARD_TASK, batch_id=batch_id)
            running_loss += loss
            running_accs += accs
            batch_id += 1
            hard_task_counter.update(list(task_idx2results['task_idx2acc'].
                keys()))
        if batch_id > config['max_train_steps']:
            break
        if (batch_id / STAGE_NUM + 1) % PER_TRAIN_LOG == 0:
            training_loss = running_loss / PER_TRAIN_LOG / STAGE_NUM
            print('Task Batch[{}]: loss_q: {:.6f}, training accs: {}'.
                format(batch_id + 1, training_loss, running_accs /
                PER_TRAIN_LOG / STAGE_NUM))
            running_loss = 0
            running_accs = np.zeros(config['update_step'] + 1)
        if (batch_id / STAGE_NUM + 1) % PER_TEST_LOG == 0:
            meta_model.eval()
            print('=== Valid tasks ===')
            valid_scores, valid_score_weights = evaluate(evaluate_generator
                (root_path, id_emb_path, mvalid_tasks, few_num=None,
                neg_num=100), meta_model, metric, device)
            avg_valid_scores = np.average(valid_scores, axis=0, weights=\
                valid_score_weights)
            print('Average valid scores: {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.
                format(*avg_valid_scores))
            print('=== Test tasks ===')
            test_scores, test_score_weights = evaluate(evaluate_generator(
                root_path, id_emb_path, mtest_tasks, few_num=None, neg_num=\
                100, is_test=True), meta_model, metric, device,
                init_compare=INIT_COMPARE)
            avg_test_scores = np.average(test_scores, axis=0, weights=\
                test_score_weights)
            print('Average test scores: {:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.
                format(*avg_test_scores))
            valid_tot_score = np.mean(avg_valid_scores)
            if valid_tot_score > best_scores:
                best_scores = valid_tot_score
                dict_save_path = os.path.join(meta_path + 'model_save/', 
                    str(batch_id + 1) + '.dict')
                # paddle.save(filter_statedict(meta_model), dict_save_path)
                paddle.save(meta_model.state_dict(), dict_save_path)
                logger.info('Best metrics: {}! Save model to {}'.format(
                    valid_tot_score, dict_save_path))
            scheduler.step(valid_tot_score)
            meta_model.train()
            print(hard_task_counter)
            # torch2paddle.invalid()


if __name__ == '__main__':
    id_emb_path = root_path + 'id_embs/'

    def poitype_pkl():
        for root, dirs, files in os.walk(root_path + 'pkls/'):
            for filename in files:
                if 'poitype_to_id.pkl' in filename:
                    return str(os.path.join(root, filename))
        print('Please check data/pkls/ ...')
        exit(2)
    poi_type2idx_path = poitype_pkl()
    POIID_DIM = 50  # the dimension of poi id embedding, related to the preprocessed embedding files in id_emb_path
    WITH_CONT_FEAT = True  # always True
    SCRATCH_ID_EMB = False  # always False

    # Settings for CHAML
    CURRICULUM = True
    HARD_TASK = True
    HARD_USER = True
    PACING_FUNCTION = 'ssp'  # ssp: single step pacing
    if HARD_USER:
        STAGE_NUM = 2
    else:
        STAGE_NUM = 1
    PER_TRAIN_LOG = 100 // STAGE_NUM
    PER_TEST_LOG = 5000 // STAGE_NUM
    PATIENCE = 2
    INIT_COMPARE = False
    logger.info(
        'method is meta. HARD_TASK: {}, HARD_USER: {}, CURRICULUM: {}, PACING_FUNCTION: {}, PER_TEST_LOG: {}, PATIENCE: {}'
        .format(HARD_TASK, HARD_USER, CURRICULUM, PACING_FUNCTION,
        PER_TEST_LOG, PATIENCE))
    logger.info('curriculum is: {}'.format(get_curriculum(root_path)))
    main_meta(meta_path, root_path, id_emb_path)
