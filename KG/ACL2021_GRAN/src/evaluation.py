#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Evaluation script."""

import json
import collections
import numpy as np

from reader.data_reader import read_examples


def generate_ground_truth(ground_truth_path, vocabulary, max_arity,
                          max_seq_length):
    """
    Generate ground truth for filtered evaluation.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, " \
        "and max_aux attribute-value pairs."

    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    all_examples, _ = read_examples(ground_truth_path)
    for (example_id, example) in enumerate(all_examples):
        # get padded input tokens and ids
        rht = [example.relation, example.head, example.tail]
        aux_attributes = []
        aux_values = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_attributes.append(attribute)
                    aux_values.append(value)

        while len(aux_attributes) < max_aux:
            aux_attributes.append("[PAD]")
            aux_values.append("[PAD]")
        assert len(aux_attributes) == max_aux
        assert len(aux_values) == max_aux

        input_tokens = rht + aux_attributes + aux_values
        input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
        assert len(input_tokens) == max_seq_length
        assert len(input_ids) == max_seq_length

        # get target answer for each pos and the corresponding key
        for pos in range(max_seq_length):
            if input_ids[pos] == 0:
                continue
            key = " ".join([
                str(input_ids[x]) for x in range(max_seq_length) if x != pos
            ])
            gt_dict[pos][key].append(input_ids[pos])

    return gt_dict


def batch_evaluation(global_idx, batch_results, all_features, gt_dict):
    """
    Perform batch evaluation.
    """
    ret_ranks = {
        'entity': [],
        'relation': [],
        '2-r': [],
        '2-ht': [],
        'n-r': [],
        'n-ht': [],
        'n-a': [],
        'n-v': []
    }
    for i, result in enumerate(batch_results):
        feature = all_features[global_idx + i]
        target = feature.mask_label
        pos = feature.mask_position
        key = " ".join([
            str(feature.input_ids[x]) for x in range(len(feature.input_ids))
            if x != pos
        ])

        # filtered setting
        rm_idx = gt_dict[pos][key]
        rm_idx = [x for x in rm_idx if x != target]
        for x in rm_idx:
            result[x] = -np.Inf
        sortidx = np.argsort(result)[::-1]

        if feature.mask_type == 1:
            ret_ranks['entity'].append(np.where(sortidx == target)[0][0] + 1)
        elif feature.mask_type == -1:
            ret_ranks['relation'].append(np.where(sortidx == target)[0][0] + 1)
        else:
            raise ValueError("Invalid `feature.mask_type`.")

        if feature.arity == 2:
            if pos == 0:
                ret_ranks['2-r'].append(np.where(sortidx == target)[0][0] + 1)
            elif pos == 1 or pos == 2:
                ret_ranks['2-ht'].append(np.where(sortidx == target)[0][0] + 1)
            else:
                raise ValueError("Invalid `feature.mask_position`.")
        elif feature.arity > 2:
            if pos == 0:
                ret_ranks['n-r'].append(np.where(sortidx == target)[0][0] + 1)
            elif pos == 1 or pos == 2:
                ret_ranks['n-ht'].append(np.where(sortidx == target)[0][0] + 1)
            elif pos > 2 and feature.mask_type == -1:
                ret_ranks['n-a'].append(np.where(sortidx == target)[0][0] + 1)
            elif pos > 2 and feature.mask_type == 1:
                ret_ranks['n-v'].append(np.where(sortidx == target)[0][0] + 1)
            else:
                raise ValueError("Invalid `feature.mask_position`.")
        else:
            raise ValueError("Invalid `feature.arity`.")

    ent_ranks = np.asarray(ret_ranks['entity'])
    rel_ranks = np.asarray(ret_ranks['relation'])
    _2_r_ranks = np.asarray(ret_ranks['2-r'])
    _2_ht_ranks = np.asarray(ret_ranks['2-ht'])
    _n_r_ranks = np.asarray(ret_ranks['n-r'])
    _n_ht_ranks = np.asarray(ret_ranks['n-ht'])
    _n_a_ranks = np.asarray(ret_ranks['n-a'])
    _n_v_ranks = np.asarray(ret_ranks['n-v'])

    return ent_ranks, rel_ranks, _2_r_ranks, _2_ht_ranks, \
           _n_r_ranks, _n_ht_ranks, _n_a_ranks, _n_v_ranks


def compute_metrics(ent_lst, rel_lst, _2_r_lst, _2_ht_lst, _n_r_lst, _n_ht_lst,
                    _n_a_lst, _n_v_lst, eval_result_file):
    """
    Combine the ranks from batches into final metrics.
    """
    all_ent_ranks = np.array(ent_lst).ravel()
    all_rel_ranks = np.array(rel_lst).ravel()
    _2_r_ranks = np.array(_2_r_lst).ravel()
    _2_ht_ranks = np.array(_2_ht_lst).ravel()
    _n_r_ranks = np.array(_n_r_lst).ravel()
    _n_ht_ranks = np.array(_n_ht_lst).ravel()
    _n_a_ranks = np.array(_n_a_lst).ravel()
    _n_v_ranks = np.array(_n_v_lst).ravel()
    all_r_ranks = np.array(_2_r_lst + _n_r_lst).ravel()
    all_ht_ranks = np.array(_2_ht_lst + _n_ht_lst).ravel()

    mrr_ent = np.mean(1.0 / all_ent_ranks)
    hits1_ent = np.mean(all_ent_ranks <= 1.0)
    hits3_ent = np.mean(all_ent_ranks <= 3.0)
    hits5_ent = np.mean(all_ent_ranks <= 5.0)
    hits10_ent = np.mean(all_ent_ranks <= 10.0)

    mrr_rel = np.mean(1.0 / all_rel_ranks)
    hits1_rel = np.mean(all_rel_ranks <= 1.0)
    hits3_rel = np.mean(all_rel_ranks <= 3.0)
    hits5_rel = np.mean(all_rel_ranks <= 5.0)
    hits10_rel = np.mean(all_rel_ranks <= 10.0)

    mrr_2r = np.mean(1.0 / _2_r_ranks)
    hits1_2r = np.mean(_2_r_ranks <= 1.0)
    hits3_2r = np.mean(_2_r_ranks <= 3.0)
    hits5_2r = np.mean(_2_r_ranks <= 5.0)
    hits10_2r = np.mean(_2_r_ranks <= 10.0)

    mrr_2ht = np.mean(1.0 / _2_ht_ranks)
    hits1_2ht = np.mean(_2_ht_ranks <= 1.0)
    hits3_2ht = np.mean(_2_ht_ranks <= 3.0)
    hits5_2ht = np.mean(_2_ht_ranks <= 5.0)
    hits10_2ht = np.mean(_2_ht_ranks <= 10.0)

    mrr_nr = np.mean(1.0 / _n_r_ranks)
    hits1_nr = np.mean(_n_r_ranks <= 1.0)
    hits3_nr = np.mean(_n_r_ranks <= 3.0)
    hits5_nr = np.mean(_n_r_ranks <= 5.0)
    hits10_nr = np.mean(_n_r_ranks <= 10.0)

    mrr_nht = np.mean(1.0 / _n_ht_ranks)
    hits1_nht = np.mean(_n_ht_ranks <= 1.0)
    hits3_nht = np.mean(_n_ht_ranks <= 3.0)
    hits5_nht = np.mean(_n_ht_ranks <= 5.0)
    hits10_nht = np.mean(_n_ht_ranks <= 10.0)

    mrr_na = np.mean(1.0 / _n_a_ranks)
    hits1_na = np.mean(_n_a_ranks <= 1.0)
    hits3_na = np.mean(_n_a_ranks <= 3.0)
    hits5_na = np.mean(_n_a_ranks <= 5.0)
    hits10_na = np.mean(_n_a_ranks <= 10.0)

    mrr_nv = np.mean(1.0 / _n_v_ranks)
    hits1_nv = np.mean(_n_v_ranks <= 1.0)
    hits3_nv = np.mean(_n_v_ranks <= 3.0)
    hits5_nv = np.mean(_n_v_ranks <= 5.0)
    hits10_nv = np.mean(_n_v_ranks <= 10.0)

    mrr_r = np.mean(1.0 / all_r_ranks)
    hits1_r = np.mean(all_r_ranks <= 1.0)
    hits3_r = np.mean(all_r_ranks <= 3.0)
    hits5_r = np.mean(all_r_ranks <= 5.0)
    hits10_r = np.mean(all_r_ranks <= 10.0)

    mrr_ht = np.mean(1.0 / all_ht_ranks)
    hits1_ht = np.mean(all_ht_ranks <= 1.0)
    hits3_ht = np.mean(all_ht_ranks <= 3.0)
    hits5_ht = np.mean(all_ht_ranks <= 5.0)
    hits10_ht = np.mean(all_ht_ranks <= 10.0)

    eval_result = {
        'entity': {
            'mrr': mrr_ent,
            'hits1': hits1_ent,
            'hits3': hits3_ent,
            'hits5': hits5_ent,
            'hits10': hits10_ent
        },
        'relation': {
            'mrr': mrr_rel,
            'hits1': hits1_rel,
            'hits3': hits3_rel,
            'hits5': hits5_rel,
            'hits10': hits10_rel
        },
        'ht': {
            'mrr': mrr_ht,
            'hits1': hits1_ht,
            'hits3': hits3_ht,
            'hits5': hits5_ht,
            'hits10': hits10_ht
        },
        '2-ht': {
            'mrr': mrr_2ht,
            'hits1': hits1_2ht,
            'hits3': hits3_2ht,
            'hits5': hits5_2ht,
            'hits10': hits10_2ht
        },
        'n-ht': {
            'mrr': mrr_nht,
            'hits1': hits1_nht,
            'hits3': hits3_nht,
            'hits5': hits5_nht,
            'hits10': hits10_nht
        },
        'r': {
            'mrr': mrr_r,
            'hits1': hits1_r,
            'hits3': hits3_r,
            'hits5': hits5_r,
            'hits10': hits10_r
        },
        '2-r': {
            'mrr': mrr_2r,
            'hits1': hits1_2r,
            'hits3': hits3_2r,
            'hits5': hits5_2r,
            'hits10': hits10_2r
        },
        'n-r': {
            'mrr': mrr_nr,
            'hits1': hits1_nr,
            'hits3': hits3_nr,
            'hits5': hits5_nr,
            'hits10': hits10_nr
        },
        'n-a': {
            'mrr': mrr_na,
            'hits1': hits1_na,
            'hits3': hits3_na,
            'hits5': hits5_na,
            'hits10': hits10_na
        },
        'n-v': {
            'mrr': mrr_nv,
            'hits1': hits1_nv,
            'hits3': hits3_nv,
            'hits5': hits5_nv,
            'hits10': hits10_nv
        },
    }
    with open(eval_result_file, "w") as fw:
        fw.write(json.dumps(eval_result, indent=4) + "\n")

    return eval_result
