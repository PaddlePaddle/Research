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
"""Model for classifier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import time
import logging
import numpy as np

from scipy.stats import pearsonr, spearmanr
from six.moves import xrange
import paddle.fluid as fluid

from model.ernie import ErnieModel

log = logging.getLogger(__name__)

def create_model(args,
                 pyreader_name,
                 ernie_config,
                 batch_size=16,
                 is_prediction=False,
                 task_name="",
                 fleet_handle=None):
    print ("DEBUG:\tclassify")
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[batch_size, args.q_max_seq_len, 1], [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.q_max_seq_len, 1], [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1],
            [batch_size, 1], [batch_size, 1]],
        dtypes=['int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64'],
        lod_levels=[0, 0, 0, 0, 0,   0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0],
        name=task_name + "_" + pyreader_name,
        use_double_buffer=True)

    (src_ids_q, sent_ids_q, pos_ids_q, task_ids_q, input_mask_q,
     src_ids_p_pos, sent_ids_p_pos, pos_ids_p_pos, task_ids_p_pos, input_mask_p_pos,
     src_ids_p_neg, sent_ids_p_neg, pos_ids_p_neg, task_ids_p_neg, input_mask_p_neg,
     labels, qids) = fluid.layers.read_file(pyreader)

    ernie_q = ErnieModel(
        src_ids=src_ids_q,
        position_ids=pos_ids_q,
        sentence_ids=sent_ids_q,
        task_ids=task_ids_q,
        input_mask=input_mask_q,
        config=ernie_config,
        model_name='query_')
    ## pos para
    ernie_pos = ErnieModel(
        src_ids=src_ids_p_pos,
        position_ids=pos_ids_p_pos,
        sentence_ids=sent_ids_p_pos,
        task_ids=task_ids_p_pos,
        input_mask=input_mask_p_pos,
        config=ernie_config,
        model_name='titlepara_')
    ## neg para
    ernie_neg = ErnieModel(
        src_ids=src_ids_p_neg,
        position_ids=pos_ids_p_neg,
        sentence_ids=sent_ids_p_neg,
        task_ids=task_ids_p_neg,
        input_mask=input_mask_p_neg,
        config=ernie_config,
        model_name='titlepara_')

    q_cls_feats = ernie_q.get_cls_output()
    pos_cls_feats = ernie_pos.get_cls_output()
    neg_cls_feats = ernie_neg.get_cls_output()
    #src_ids_p_pos = fluid.layers.Print(src_ids_p_pos, message='pos: ')
    #pos_cls_feats = fluid.layers.Print(pos_cls_feats, message='pos: ')

    p_cls_feats = fluid.layers.concat([pos_cls_feats, neg_cls_feats], axis=0)

    if is_prediction:
        p_cls_feats = fluid.layers.slice(p_cls_feats, axes=[0], starts=[0], ends=[batch_size])
        multi = fluid.layers.elementwise_mul(q_cls_feats, p_cls_feats)
        probs = fluid.layers.reduce_sum(multi, dim=-1)

        graph_vars = {
            "probs": probs,
            "qids": qids,
            "q_rep": q_cls_feats,
            "p_rep": p_cls_feats
        }
        return pyreader, graph_vars

    if args.use_cross_batch and fleet_handle is not None:
        print("worker num is: {}".format(fleet_handle.worker_num()))
        all_p_cls_feats = fluid.layers.collective._c_allgather(
                p_cls_feats, fleet_handle.worker_num(), use_calc_stream=True)

        #multiply
        logits = fluid.layers.matmul(q_cls_feats, all_p_cls_feats, transpose_x=False, transpose_y=True)
        worker_id = fleet_handle.worker_index()

    else:
        logits = fluid.layers.matmul(q_cls_feats, p_cls_feats, transpose_x=False, transpose_y=True)
        worker_id = 0

    probs = logits

    all_labels = np.array(range(batch_size * worker_id * 2, batch_size * (worker_id * 2 + 1)), dtype='int64')
    matrix_labels = fluid.layers.assign(all_labels)
    matrix_labels = fluid.layers.unsqueeze(matrix_labels, axes=1)
    matrix_labels.stop_gradient=True
#    fluid.layers.Print(matrix_labels, message='matrix_labels')

    #print('DEBUG:\tstart loss')
    ce_loss = fluid.layers.softmax_with_cross_entropy(
           logits=logits, label=matrix_labels)
    loss = fluid.layers.mean(x=ce_loss)
    #print('DEBUG:\tloss done')

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(
        input=probs, label=matrix_labels)

    graph_vars = {
        "loss": loss,
        "probs": probs,
        "accuracy": accuracy,
        "labels": labels,
        "num_seqs": num_seqs,
        "qids": qids,
        "q_rep": q_cls_feats,
        "p_rep": p_cls_feats
    }

    cp = []
    cp.extend(ernie_q.checkpoints)
    cp.extend(ernie_pos.checkpoints)
    cp.extend(ernie_neg.checkpoints)
    return pyreader, graph_vars, cp


def evaluate_mrr(preds):
    last_qid = None
    total_mrr = 0.0
    qnum = 0.0
    rank = 0.0
    correct = False
    for qid, score, label in preds:
        if qid != last_qid:
            rank = 0.0
            qnum += 1
            correct = False
            last_qid = qid

        rank += 1
        if not correct and label != 0:
            total_mrr += 1.0 / rank
            correct = True

    return total_mrr / qnum


def evaluate_map(preds):
    def singe_map(st, en):
        total_p = 0.0
        correct_num = 0.0
        for index in xrange(st, en):
            if int(preds[index][2]) != 0:
                correct_num += 1
                total_p += correct_num / (index - st + 1)
        if int(correct_num) == 0:
            return 0.0
        return total_p / correct_num

    last_qid = None
    total_map = 0.0
    qnum = 0.0
    st = 0
    for i in xrange(len(preds)):
        qid = preds[i][0]
        if qid != last_qid:
            qnum += 1
            if last_qid != None:
                total_map += singe_map(st, i)
            st = i
            last_qid = qid

    total_map += singe_map(st, len(preds))
    return total_map / qnum


def evaluate(exe,
                      test_program,
                      test_pyreader,
                      graph_vars,
                      eval_phase,
                      use_multi_gpu_test=False,
                      metric='simple_accuracy'):
    train_fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["num_seqs"].name
    ]

    if eval_phase == "train":
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list, program=test_program)
        ret = {"loss": np.mean(outputs[0]), "accuracy": np.mean(outputs[1])}
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[3][0])
        return ret

    test_pyreader.start()
    total_cost, total_acc, total_num_seqs, total_label_pos_num, total_pred_pos_num, total_correct_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    qids, labels, scores, preds = [], [], [], []
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name, graph_vars["accuracy"].name,
        graph_vars["probs"].name, graph_vars["labels"].name,
        graph_vars["num_seqs"].name, graph_vars["qids"].name,
        graph_vars["q_rep"].name, graph_vars["p_rep"].name
    ]
    #emb_file = open('emb_qp', 'w')
    while True:
        try:
            if use_multi_gpu_test:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids, q_rep, p_rep = exe.run(
                    fetch_list=fetch_list)
            else:
                np_loss, np_acc, np_probs, np_labels, np_num_seqs, np_qids, q_rep, p_rep = exe.run(
                    program=test_program, fetch_list=fetch_list)
            total_cost += np.sum(np_loss * np_num_seqs)
            total_acc += np.sum(np_acc * np_num_seqs)
            total_num_seqs += np.sum(np_num_seqs)
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            batch_scores = np.diag(np_probs).reshape(-1).tolist()
            scores.extend(batch_scores)
            #for item in list(zip(q_rep, p_rep, batch_scores)):
            #    _left = ' '.join([str(each) for each in item[0]])
            #    _right = ' '.join([str(each) for each in item[1]])
            #    emb_file.write(_left + '\t' + _right + '\t' + str(item[2]) + '\n')
            #scores.extend(np_probs[:, 1].reshape(-1).tolist())
            #np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            #preds.extend(np_preds)
            #total_label_pos_num += np.sum(np_labels)
            #total_pred_pos_num += np.sum(np_preds)
            #total_correct_num += np.sum(np.dot(np_preds, np_labels))
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    #for score in np_preds:
    #    print (score)
    #print ('---------------------')
    #time_end = time.time()
    #cost = total_cost / total_num_seqs
    #elapsed_time = time_end - time_begin
    #emb_file.close()
    return None
    evaluate_info = ""
    if metric == 'acc_and_f1':
        ret = acc_and_f1(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, ave_acc: %f, f1: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret['acc'], ret['f1'], total_num_seqs, elapsed_time)
    elif metric == 'matthews_corrcoef':
        ret = matthews_corrcoef(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, matthews_corrcoef: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret, total_num_seqs, elapsed_time)
    elif metric == 'pearson_and_spearman':
        ret = pearson_and_spearman(scores, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, pearson:%f, spearman:%f, corr:%f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret['pearson'], ret['spearman'], ret['corr'], total_num_seqs, elapsed_time)
    elif metric == 'simple_accuracy':
        ret = simple_accuracy(preds, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, acc:%f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret, total_num_seqs, elapsed_time)
    elif metric == "acc_and_f1_and_mrr":
        ret_a = acc_and_f1(preds, labels)
        preds = sorted(
            zip(qids, scores, labels), key=lambda elem: (elem[0], -elem[1]))
        ret_b = evaluate_mrr(preds)
        evaluate_info = "[%s evaluation] ave loss: %f, acc: %f, f1: %f, mrr: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, cost, ret_a['acc'], ret_a['f1'], ret_b, total_num_seqs, elapsed_time)
    else:
        raise ValueError('unsupported metric {}'.format(metric))
    return evaluate_info


def matthews_corrcoef(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc


def f1_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return f1


def pearson_and_spearman(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def acc_and_f1(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    acc = simple_accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def predict(exe,
            test_program,
            test_pyreader,
            graph_vars,
            dev_count=1):
    test_pyreader.start()
    qids, scores, probs = [], [], []
    preds = []

    fetch_list = [graph_vars["probs"].name, graph_vars["qids"].name, \
                  graph_vars["q_rep"].name, graph_vars["p_rep"].name,]

    emb_file = open('emb_qp', 'w')
    while True:
        try:
            if dev_count == 1:
                np_probs, np_qids, q_rep, p_rep = exe.run(program=test_program,
                                            fetch_list=fetch_list)
            else:
                np_probs, np_qids, q_rep, p_rep = exe.run(fetch_list=fetch_list)

            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            batch_scores = np_probs.reshape(-1).tolist()
            for item in list(zip(q_rep, p_rep, batch_scores)):
                _left = ' '.join([str(each) for each in item[0]])
                _right = ' '.join([str(each) for each in item[1]])
                #emb_file.write(_left + '\t' + _right + '\t' + str(item[2]) + '\n')
                #emb_file.write(_right + '\n')
                emb_file.write(str(item[2]) + '\n')
            #for score in batch_scores:
            #    print (score)
            #print ('--------')
            #if is_classify:
            #    np_preds = np.argmax(np_probs, axis=1).astype(np.float32)
            #    preds.extend(np_preds)
            #elif is_regression:
            #    preds.extend(np_probs.reshape(-1))

            probs.extend(batch_scores)

        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    emb_file.close()
    #probs = np.concatenate(probs, axis=0).reshape([len(preds), -1])

    return qids, preds, probs
