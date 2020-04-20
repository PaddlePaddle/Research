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

import logging
import time

import numpy as np
import paddle.fluid as fluid
from model.ernie import ErnieModel
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from six.moves import xrange

log = logging.getLogger(__name__)


def cls_from_ernie(
    args,
    src_ids,
    position_ids,
    sentence_ids,
    task_ids,
    input_mask,
    config,
    use_fp16,
):
    """cls_from_ernie"""
    ernie = ErnieModel(
        src_ids=src_ids,
        position_ids=position_ids,
        sentence_ids=sentence_ids,
        task_ids=task_ids,
        input_mask=input_mask,
        config=config,
        use_fp16=use_fp16,
    )
    cls_feats = ernie.get_pooled_output()
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train",
    )
    return cls_feats


def create_model(
    args,
    pyreader_name,
    ernie_config,
    is_prediction=False,
    task_name="",
    is_classify=False,
    is_regression=False,
    ernie_version="1.0",
):
    """create_model"""
    if is_classify:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1], [-1, args.max_seq_len, 1],
                [-1, args.max_seq_len, 1],
                [-1, 1], [-1, 1], [-1, 1],
            ],
            dtypes=[
                'int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64', 'int64', 'int64', 'float32',
                'int64', 'int64', 'int64',
            ],
            lod_levels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            name=task_name + "_" + pyreader_name,
            use_double_buffer=True,
        )

    (
        src_ids_1, sent_ids_1, pos_ids_1, task_ids_1, input_mask_1,
        src_ids_2, sent_ids_2, pos_ids_2, task_ids_2, input_mask_2,
        src_ids_3, sent_ids_3, pos_ids_3, task_ids_3, input_mask_3,
        labels, types, qids,
    ) = fluid.layers.read_file(pyreader)

    cls_feats_query = cls_from_ernie(
        args,
        src_ids=src_ids_1,
        position_ids=pos_ids_1,
        sentence_ids=sent_ids_1,
        task_ids=task_ids_1,
        input_mask=input_mask_1,
        config=ernie_config,
        use_fp16=args.use_fp16,
    )

    cls_feats_left = cls_from_ernie(
        args,
        src_ids=src_ids_2,
        position_ids=pos_ids_2,
        sentence_ids=sent_ids_2,
        task_ids=task_ids_2,
        input_mask=input_mask_2,
        config=ernie_config,
        use_fp16=args.use_fp16,
    )

    cls_feats_right = cls_from_ernie(
        args,
        src_ids=src_ids_3,
        position_ids=pos_ids_3,
        sentence_ids=sent_ids_3,
        task_ids=task_ids_3,
        input_mask=input_mask_3,
        config=ernie_config,
        use_fp16=args.use_fp16,
    )

    left_concat = fluid.layers.concat(
        input=[cls_feats_query, cls_feats_left], axis=-1,
    )

    right_concat = fluid.layers.concat(
        input=[cls_feats_query, cls_feats_right], axis=-1,
    )

    left_score = fluid.layers.fc(
        input=left_concat,
        size=1,
        param_attr=fluid.ParamAttr(
            name=task_name + "_cls_out_w_left",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02),
        ),
        bias_attr=fluid.ParamAttr(
            name=task_name + "_cls_out_b_left",
            initializer=fluid.initializer.Constant(0.),
        ),
    )
    left_score = fluid.layers.sigmoid(left_score)

    right_score = fluid.layers.fc(
        input=right_concat,
        size=1,
        param_attr=fluid.ParamAttr(
            name=task_name + "_cls_out_w_right",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02),
        ),
        bias_attr=fluid.ParamAttr(
            name=task_name + "_cls_out_b_right",
            initializer=fluid.initializer.Constant(0.),
        ),
    )
    right_score = fluid.layers.sigmoid(right_score)

    type_out = fluid.layers.fc(
        input=cls_feats_query,
        size=24,
        param_attr=fluid.ParamAttr(
            name=task_name + "_cls_out_w_type",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02),
        ),
        bias_attr=fluid.ParamAttr(
            name=task_name + "_cls_out_b_type",
            initializer=fluid.initializer.Constant(0.),
        ),
    )

    if is_prediction:
        left_probs = left_score
        right_probs = right_score

        type_probs = fluid.layers.softmax(type_out)
        feed_targets_name = [
            src_ids_1.name, sent_ids_1.name, pos_ids_1.name, task_ids_1.name, input_mask_1.name,
            src_ids_2.name, sent_ids_2.name, pos_ids_2.name, task_ids_2.name, input_mask_2.name,
            src_ids_3.name, sent_ids_3.name, pos_ids_3.name, task_ids_3.name, input_mask_3.name,
            qids.name,
        ]
        ret = {}
        ret['pyreader'] = pyreader
        ret['left_probs'] = left_probs
        ret['right_probs'] = right_probs
        ret['type_probs'] = type_probs
        ret['feed_targets_name'] = feed_targets_name
        return ret

    num_seqs = fluid.layers.create_tensor(dtype='int64')
    labels = fluid.layers.cast(x=labels, dtype="float32")
    types = fluid.layers.cast(x=types, dtype="int64")
    label_loss = fluid.layers.rank_loss(
        label=labels, left=left_score, right=right_score,
    )
    type_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=type_out, label=types, return_softmax=True,
    )

    loss = fluid.layers.mean(x=label_loss) + fluid.layers.mean(x=type_loss)
    graph_vars = {
        "loss": loss,
        "left_score": left_score,
        "right_score": right_score,
        "labels": labels,
        "probs": probs,
        "types": types,
        "num_seqs": num_seqs,
        "qids": qids,
    }

    return pyreader, graph_vars


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
            if last_qid is not None:
                total_map += singe_map(st, i)
            st = i
            last_qid = qid

    total_map += singe_map(st, len(preds))
    return total_map / qnum


def evaluate(
    exe,
    test_program,
    test_pyreader,
    graph_vars,
    eval_phase,
    use_multi_gpu_test=False,
    metric='simple_accuracy',
    is_classify=False,
    is_regression=False,
):
    train_fetch_list = [
        graph_vars["loss"].name, graph_vars["left_score"].name,
        graph_vars["right_score"].name, graph_vars["labels"].name,
    ]

    if eval_phase == "train":
        if "learning_rate" in graph_vars:
            train_fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=train_fetch_list)
        left_score = outputs[1]
        right_score = outputs[2]
        label = [i[0] for i in outputs[3]]
        acc_and_f1_res = acc_and_f1(left_score, right_score, label)

        ret = {
            "loss": np.mean(outputs[0]), "left_score": np.mean(outputs[1]),
            "right_score": np.mean(outputs[2]), 'acc': acc_and_f1_res['acc'], 'f1': acc_and_f1_res['f1'],
        }
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[4][0])
        return ret

    test_pyreader.start()
    total_cost, total_acc, total_num_seqs, total_label_pos_num, total_pred_pos_num, total_correct_num = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    qids, labels, left_score, right_score, loss = [], [], [], [], []
    time_begin = time.time()

    fetch_list = [
        graph_vars["loss"].name, graph_vars["left_score"].name,
        graph_vars["right_score"].name, graph_vars["labels"].name,
        graph_vars["qids"].name,
    ]
    while True:
        try:
            if use_multi_gpu_test:
                np_loss, np_left_score, np_right_score, np_labels, np_qids = exe.run(
                    fetch_list=fetch_list,
                )
            else:
                np_loss, np_left_score, np_right_score, np_labels, np_qids = exe.run(
                    program=test_program, fetch_list=fetch_list,
                )
            labels.extend(np_labels.reshape((-1)).tolist())
            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            left_score.extend(np_left_score.reshape((-1)).tolist())
            right_score.extend(np_right_score.reshape((-1)).tolist())
            loss.extend(np_loss.reshape((-1)).tolist())
        except fluid.core.EOFException:
            test_pyreader.reset()
            break
    time_end = time.time()
    elapsed_time = time_end - time_begin

    evaluate_info = ""
    if metric == 'acc_and_f1':
        ret = acc_and_f1(left_score, right_score, labels)
        evaluate_info = "[%s evaluation] ave loss: %f, ave_acc: %f, f1: %f, data_num: %d, elapsed time: %f s" \
            % (eval_phase, np.mean(np.array(loss)), ret['acc'], ret['f1'], 0, elapsed_time)
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
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn),
    )
    return mcc


def f1_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp * 1.0 / (tp + fp)
    r = tp * 1.0 / (tp + fn)
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


def acc_and_f1(left_score, right_score, labels):
    left_score = np.array(left_score)
    right_score = np.array(right_score)
    preds = []
    for i in range(len(left_score)):
        if left_score[i] > right_score[i]:
            preds.append(1)
        else:
            preds.append(0)
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


def predict(
    exe,
    test_program,
    test_pyreader,
    graph_vars,
    dev_count=1,
    is_classify=False,
    is_regression=False,
):
    test_pyreader.start()
    qids, left_scores, right_scores = [], [], []
    preds = []

    fetch_list = [
        graph_vars["left_score"].name,
        graph_vars["right_score"].name, graph_vars["qids"].name,
    ]

    while True:
        try:
            if dev_count == 1:
                np_left_score, np_right_score, np_qids = exe.run(
                    program=test_program,
                    fetch_list=fetch_list,
                )
            else:
                np_left_score, np_right_score, np_qids = exe.run(
                    fetch_list=fetch_list,
                )

            if np_qids is None:
                np_qids = np.array([])
            qids.extend(np_qids.reshape(-1).tolist())
            left_scores.extend(np_left_score.reshape(-1).tolist())
            right_scores.extend(np_right_score.reshape(-1).tolist())

        except fluid.core.EOFException:
            test_pyreader.reset()
            break

    return qids, left_scores, right_scores
