#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Mask, padding and batching."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import copy


def mask(batch_tokens, seg_labels, total_token_num, vocab_size, CLS=0, SEP=2, MASK=50264):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []

    # bidirectional mask language model
    prob_mask = np.random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    replace_ids = np.random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        prob_index += pre_sent_len
        beg = 0
        for token_index, token in enumerate(sent):
            seg_label = seg_labels[sent_index][token_index]
            if seg_label == 1:
                continue
            if beg == 0:
                if seg_label != -1:
                    beg = token_index
                continue

            prob = prob_mask[prob_index + beg]
            if prob > 0.15:
                pass
            else:
                for index in range(beg, token_index):
                    prob = prob_mask[prob_index + index]
                    base_prob = 1.0
                    if index == beg:
                        base_prob = 0.15
                    if base_prob * 0.2 < prob <= base_prob:
                        mask_label.append(sent[index])
                        sent[index] = MASK
                        mask_pos.append(sent_index * max_len + index)
                    elif base_prob * 0.1 < prob <= base_prob * 0.2:
                        mask_label.append(sent[index])
                        sent[index] = replace_ids[prob_index + index]
                        mask_pos.append(sent_index * max_len + index)
                    else:
                        mask_label.append(sent[index])
                        mask_pos.append(sent_index * max_len + index)

            if seg_label == -1:
                beg = 0
            else:
                beg = token_index
        pre_sent_len = len(sent)
    if len(mask_label) == 0:
        mask_label.append(sent[1])
        mask_pos.append(1)
    mask_label = np.array(mask_label).astype('int64').reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype('int64').reshape([-1, 1])
    return batch_tokens, mask_label, mask_pos


def gen_phrase_samples(dp_ids, max_neg_num):
    """Generate phrase samples for batch data."""
    max_len = max([len(sent) for sent in dp_ids])
    batch_samples = []
    batch_positives = []
    batch_negatives = []
    batch_negatives_mask = []
    for sent_index, dp_id in enumerate(dp_ids):
        deptree = DepTree(dp_id)
        samples, positives, negatives = deptree('phrase', max_neg_num)
        assert len(samples) == len(positives) == len(negatives), "samples, positives, negatives should have same length"
        if not samples:
            continue
        samples = np.array(samples)
        positives = np.array(positives)

        # padding
        samples = samples + sent_index * max_len
        positives = positives + sent_index * max_len

        batch_samples.extend(samples)
        batch_positives.extend(positives)
        for negs in negatives:
            negs_mask = len(negs) * [1] + (max_neg_num - len(negs)) * [0]
            negs += (max_neg_num - len(negs)) * [0]
            negs = [neg + sent_index * max_len for neg in negs]
            batch_negatives.append(negs)
            batch_negatives_mask.append(negs_mask)
    if not batch_samples:
        # if batch samples is [], generate a fake sample
        batch_samples.append(0)
        batch_positives.append(0)
        batch_negatives.append([0] * max_neg_num)
        batch_negatives_mask.append([1] + [0] * (max_neg_num - 1))
    batch_samples = np.array(batch_samples).astype('int64').reshape([-1, 1])
    batch_positives = np.array(batch_positives).astype('int64').reshape([-1, 1])
    batch_negatives = np.array(batch_negatives).astype('int64').reshape([-1, max_neg_num, 1])
    batch_negatives_mask = np.array(batch_negatives_mask).astype('float32').reshape([-1, max_neg_num])
    return batch_samples, batch_positives, batch_negatives, batch_negatives_mask


def gen_tree_samples(dp_ids, max_sub_num, max_neg_num):
    max_len = max([len(sent) for sent in dp_ids])
    batch_samples = []
    batch_positives = []
    batch_positives_mask = []
    batch_negatives = []
    batch_negatives_mask = []
    for sent_index, dp_id in enumerate(dp_ids):
        deptree = DepTree(dp_id)
        samples, positives, negatives = deptree('tree', max_sub_num, max_neg_num)
        samples = [sample + sent_index * max_len for sample in samples]
        positives = [[pos + sent_index * max_len for pos in positive] for positive in positives]
        negatives = [[[n + sent_index * max_len for n in neg] for neg in negative] for negative in negatives]
        assert len(samples) == len(positives) == len(negatives)
        if not samples:
            # if samples is [], generate a fake sample
            samples = [1]
            positives = [[1]]
            negatives = [[[1]]]
        batch_samples.extend(samples)

        pad_positives = []
        positives_mask = []
        for positive in positives:
            _mask = [1] * len(positive) + [0] * (max_sub_num - len(positive))
            positive += [0] * (max_sub_num - len(positive))
            pad_positives.append(positive)
            positives_mask.append(_mask)
        batch_positives.extend(pad_positives)
        batch_positives_mask.extend(positives_mask)

        pad_negatives = []
        negatives_mask = []
        for negs in negatives:
            pad_negs = []
            negs_mask = []
            for neg in negs:
                _mask = [1] * len(neg) + [0] * (max_sub_num + 1 - len(neg))
                neg += [0] * (max_sub_num + 1 - len(neg))
                pad_negs.append(neg)
                negs_mask.append(_mask)
            pad_negs.extend([[0] * (max_sub_num + 1)] * (max_neg_num - len(pad_negs)))
            negs_mask.extend([[0] * (max_sub_num + 1)] * (max_neg_num - len(negs_mask)))
            pad_negatives.append(pad_negs)
            negatives_mask.append(negs_mask)

        batch_negatives.extend(pad_negatives)
        batch_negatives_mask.extend(negatives_mask)

    batch_samples = np.array(batch_samples).astype('int64').reshape([-1, 1])
    batch_positives = np.array(batch_positives).astype('int64').reshape([-1, max_sub_num, 1])
    batch_positives_mask = np.array(batch_positives_mask).astype('float32').reshape([-1, max_sub_num])
    batch_negatives = np.array(batch_negatives).astype('int64').reshape([-1, max_neg_num, max_sub_num + 1, 1])
    batch_negatives_mask = np.array(batch_negatives_mask).astype('float32').reshape([-1, max_neg_num, max_sub_num + 1])
    return batch_samples, batch_positives, batch_positives_mask, batch_negatives, batch_negatives_mask


def prepare_batch_data(insts,
                       total_token_num,
                       tree_max_sub_num=10,
                       tree_max_neg_num=10,
                       phrase_max_neg_num=10,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False):

    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    seg_labels = [inst[3] for inst in insts]
    dp_ids = [inst[4] for inst in insts]

    # First step: do mask without padding
    assert mask_id >= 0, "[FATAL] mask_id must >= 0"
    out, mask_label, mask_pos = mask(batch_src_ids,
                                     seg_labels,
                                     total_token_num,
                                     vocab_size=voc_size,
                                     CLS=cls_id,
                                     SEP=sep_id,
                                     MASK=mask_id)

    phrase_batch_samples, phrase_batch_positives, phrase_batch_negatives, phrase_batch_negatives_mask = \
        gen_phrase_samples(dp_ids, phrase_max_neg_num)
    tree_batch_samples, tree_batch_positives, tree_batch_positives_mask, tree_batch_negatives, tree_batch_negatives_mask = gen_tree_samples(
        dp_ids, tree_max_sub_num, tree_max_neg_num)

    # Second step: padding
    src_id, self_input_mask = pad_batch_data(out,
                                             pad_idx=pad_id,
                                             return_input_mask=return_input_mask,
                                             return_max_len=return_max_len,
                                             return_num_token=return_num_token)

    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id)
    return_list = [
        src_id, pos_id, sent_id, self_input_mask, mask_label, mask_pos, phrase_batch_samples, phrase_batch_positives,
        phrase_batch_negatives, phrase_batch_negatives_mask, tree_batch_samples, tree_batch_positives,
        tree_batch_positives_mask, tree_batch_negatives, tree_batch_negatives_mask
    ]

    return return_list


def pad_batch_data(insts,
                   pad_idx=1,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False,
                   return_seq_lens=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype('int64').reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst)) for inst in insts])

        return_list += [inst_pos.astype('int64').reshape([-1, max_len, 1])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        input_mask_data = np.matmul(input_mask_data, np.transpose(input_mask_data, (0, 2, 1)))
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype('int64').reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


class NODE:
    """NODE class"""

    def __init__(self, id=None, parent=None):
        self.lefts = []
        self.rights = []
        self.childs = []
        self.id = int(id)
        self.parent = parent if parent is None else int(parent)
        self.des_span = (id, id)


class DepTree:
    """
    DepTree class, used to check whether the prediction result is a project Tree.
    A projective tree means that you can project the tree without crossing arcs.
    """

    def __init__(self, heads):
        # set root head to -1
        heads = copy.deepcopy(heads)
        if len(heads) == 256:
            self.max_pos = 254
        else:
            self.max_pos = 510
        heads = np.array(heads)
        heads[heads > self.max_pos] = self.max_pos
        heads[0] = -1
        self.heads = heads
        self.visit = [False] * len(heads)
        self.node_pair = []
        self.parents = None
        self.build_tree()

    def build_tree(self):
        """Build the tree"""
        self.nodes = [NODE(index, head) for index, head in enumerate(self.heads)]
        # set root
        self.root = self.nodes[0]
        node_pair = []
        parents = set()
        parents_wo_root = set()
        for node in self.nodes[1:-1]:
            self.add(self.nodes[node.parent], node)
            if node.parent != 0 and node.parent != self.max_pos:
                node_pair.append([self.nodes[node.parent], node])
                self.add_descendants(self.nodes[node.parent], node)
                parents.add(node.parent)
                if self.nodes[node.parent].parent != 0:
                    parents_wo_root.add(node.parent)
        self.node_pair = np.array(node_pair)
        self.parents = np.array(list(parents))
        self.parents_wo_root = np.array(list(parents_wo_root))

    def add_descendants(self, parent, child):
        if parent.id == 0 or parent.parent == -1 or parent.id == self.max_pos:
            return
        _min, _max = parent.des_span
        parent.des_span = (min(_min, child.id), max(_max, child.id))
        self.add_descendants(self.nodes[parent.parent], child)

    def add(self, parent: NODE, child: NODE):
        """Add a child node"""
        if parent.id is None or child.id is None:
            raise f"id is None"
        parent.childs.append(child.id)
        if parent.id < child.id:
            parent.rights = sorted(parent.rights + [child.id])
        else:
            parent.lefts = sorted(parent.lefts + [child.id])

    def gen_phrase_samples(self, max_neg_num):
        """Generate phrase samples"""
        rand = np.random.rand(len(self.parents)) < 0.15
        parents = self.parents[rand]
        node_size = len(self.nodes)
        all_samples = []
        all_positives = []
        all_negatives = []
        for parent_id in parents:
            parent = self.nodes[parent_id]
            span = parent.des_span
            assert span[0] != span[1]
            x_h = parent_id
            x_in, = self.sample_skip_span(*span, parent_id, parent_id, k=1)
            x_outs = self.sample_skip_span(1, node_size - 2, *span, k=max_neg_num, m=10)
            if not x_outs:
                continue
            # delete parent node of x_h
            new_outs = []
            for x_out in x_outs:
                if x_out != self.nodes[x_h].parent:
                    new_outs.append(x_out)
            x_outs = new_outs

            all_samples.append(x_h)
            all_positives.append(x_in)
            all_negatives.append(x_outs)

            # the distance between x_out and x_h should be greater than the distance between x_in and x_h
            re_outs = []
            for x_out in x_outs:
                if x_h < x_in < x_out or x_out < x_in < x_h:
                    re_outs.append(x_out)
            if re_outs:
                all_samples.append(x_in)
                all_positives.append(x_h)
                all_negatives.append(re_outs)
        return all_samples, all_positives, all_negatives

    def gen_tree_samples(self, max_sub_num, max_neg_num):
        """Generate tree samples"""

        def get_negatives(cur_id, start, end):
            negatives = []
            x, y = start - 1, end - 1
            # and not (x >= start and y <= end): prevent the situation that the sample is included the negative sample
            while x >= 1 and y >= cur_id and not (x >= start and y <= end):
                negatives.append(list(range(x, y + 1)))
                x -= 1
                y -= 1
            x, y = start + 1, end + 1
            while x <= cur_id and y <= min(self.max_pos, len(self.nodes) - 2) and not (x >= start and y <= end):
                negatives.append(list(range(x, y + 1)))
                x += 1
                y += 1
            return negatives

        all_samples = []
        all_positives = []
        all_negatives = []
        for parent_id in self.parents_wo_root:
            parent = self.nodes[parent_id]
            span = parent.des_span

            assert span[0] != span[1]
            # skip the node which the number of subtree nodes is greater than max_sub_num
            if span[1] - span[0] + 1 > max_sub_num:
                continue

            positive = list(range(span[0], span[1] + 1))
            negatives = []
            negatives += get_negatives(parent_id, span[0], span[1])
            negatives += get_negatives(parent_id, span[0], span[1] + 1)
            # skip the node which the number of subtree nodes is less than 3
            if span[1] - span[0] + 1 > 3:
                negatives += get_negatives(parent_id, span[0], span[1] - 1)
            if len(negatives) > max_neg_num:
                negatives = random.sample(negatives, k=max_neg_num)
            if not negatives:
                continue
            # filter negatives
            f_negatives = []
            for negative in negatives:
                if set(negative) == set(positive) | set([parent_id]):
                    continue
                f_negatives.append(negative)
            negatives = f_negatives

            all_samples.append(parent_id)
            all_positives.append(positive)
            all_negatives.append(negatives)
        return all_samples, all_positives, all_negatives

    def sample_skip_span(self, begin, end, span_b, span_e, k, m=3):
        targets = list(range(max(begin, span_b - m), span_b)) + list(range(span_e + 1, min(span_e + m, end) + 1))
        if len(targets) < 1:
            return []
        return random.sample(targets, k=min(k, len(targets)))

    def __call__(self, mode, *args, **kwargs):
        if mode == "tree":
            return self.gen_tree_samples(*args, *kwargs)
        elif mode == 'phrase':
            return self.gen_phrase_samples(*args, *kwargs)
        else:
            raise ValueError(f"error! Unknown model({mode}).")


if __name__ == "__main__":
    pass
