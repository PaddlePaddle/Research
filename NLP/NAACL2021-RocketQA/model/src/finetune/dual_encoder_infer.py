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

import faiss
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
                 task_name=""):
    pyreader = fluid.layers.py_reader(
        capacity=50,
        shapes=[[batch_size, args.q_max_seq_len, 1], [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.q_max_seq_len, 1], [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.q_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1], [batch_size, args.p_max_seq_len, 1],
            [batch_size, args.p_max_seq_len, 1],
            [batch_size, 1], [batch_size, 1]],
    dtypes=['int64', 'int64', 'int64', 'int64', 'float32',
            'int64', 'int64', 'int64', 'int64', 'float32',
            'int64', 'int64'],
    lod_levels=[0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0],
    name=pyreader_name,
    use_double_buffer=True)

    (src_ids_q, sent_ids_q, pos_ids_q, task_ids_q, input_mask_q,
     src_ids_p, sent_ids_p, pos_ids_p, task_ids_p, input_mask_p,
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
    ernie_p = ErnieModel(
        src_ids=src_ids_p,
        position_ids=pos_ids_p,
        sentence_ids=sent_ids_p,
        task_ids=task_ids_p,
        input_mask=input_mask_p,
        config=ernie_config,
        model_name='titlepara_')

    q_cls_feats = ernie_q.get_cls_output()
    p_cls_feats = ernie_p.get_cls_output()
    #p_cls_feats = fluid.layers.concat([pos_cls_feats, neg_cls_feats], axis=0)
    #src_ids_p = fluid.layers.Print(src_ids_p, message='p: ')
    #p_cls_feats = fluid.layers.Print(p_cls_feats, message='p: ')

    #multiply
    logits = fluid.layers.matmul(q_cls_feats, p_cls_feats, transpose_x=False, transpose_y=True)
    probs = logits
    #fluid.layers.Print(probs, message='probs: ')
    #logits2 = fluid.layers.elementwise_mul(x=q_rep, y=p_rep)
    #fluid.layers.Print(logits2, message='logits2: ')
    #probs2 = fluid.layers.reduce_sum(logits, dim=-1)
    #fluid.layers.Print(probs2, message='probs2: ')

    matrix_labels = fluid.layers.eye(batch_size, batch_size, dtype='float32')
    matrix_labels.stop_gradient=True

    #print('DEBUG:\tstart loss')
    ce_loss, _ = fluid.layers.softmax_with_cross_entropy(
           logits=logits, label=matrix_labels, soft_label=True, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)
    #print('DEBUG:\tloss done')

    matrix_labels = fluid.layers.argmax(matrix_labels, axis=-1)
    matrix_labels = fluid.layers.reshape(x=matrix_labels, shape=[batch_size, 1])
    num_seqs = fluid.layers.create_tensor(dtype='int64')
    accuracy = fluid.layers.accuracy(input=probs, label=matrix_labels, total=num_seqs)

    #ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
    #    logits=logits, label=labels, return_softmax=True)
    #loss = fluid.layers.mean(x=ce_loss)
    #accuracy = fluid.layers.accuracy(
    #    input=probs, label=labels, total=num_seqs)
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

    return pyreader, graph_vars

def build_engine(para_emb_list, dim):
    index = faiss.IndexFlatIP(dim)
    # add paragraph embedding
    p_emb_matrix = np.asarray(para_emb_list)
    index.add(p_emb_matrix.astype('float32'))
    #print ("insert done", file=sys.stderr)
    return index

def predict(args,
            exe,
            test_program,
            test_pyreader,
            graph_vars,
            dev_count=1,
            output_item=0,
            output_file_name='emb',
            hidden_size=768):

    test_pyreader.start()
    fetch_list = [graph_vars["q_rep"].name, graph_vars["p_rep"].name,]
    para_embs = []

    batch_id = 0
    while True:
        try:
            batch_id += 1
            if batch_id % 500 == 0:
                log.info("complete batch %s" % batch_id)
            q_rep, p_rep = exe.run(program=test_program,
                                            fetch_list=fetch_list)

            if output_item == 0:
                for item in q_rep:
                    para_embs.append(np.array(item, dtype='float32'))
            elif output_item == 1:
                for item in p_rep:
                    para_embs.append(np.array(item, dtype='float32'))

        except fluid.core.EOFException:
            test_pyreader.reset()
            break

    log.info("predict embs cnt: %s" % len(para_embs))
    para_embs = para_embs[:args.test_data_cnt]
    log.info("cut embs cnt: %s" % len(para_embs))

    if output_item == 1:
        engine = build_engine(para_embs, hidden_size)
        faiss.write_index(engine, output_file_name)
        log.info("create index done!")
    else:
        emb_matrix = np.asarray(para_embs)
        np.save(output_file_name + '.npy', emb_matrix)
        log.info("save to npy file!")
