#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

"""
File: unimo_grounded_batching.py
Author: liwei(liwei85@baidu.com)
Date: 2021-09-23 15:43
Desc: Mask, padding and batching.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import random
import math


def mask(batch_tokens,
         seg_labels,
         total_token_num,
         txt_mask_ratio=0.15,
         pair_labels=None,
         vocab_size=0,
         pretraining_task='seq2seq',
         CLS=0,
         SEP=2,
         MASK=50264,
         sent_b_starts=None,
         del_pos=None,
         geometric_p=0.2,
         span_lower=0,
         span_upper=10,
         batch_can_mask=None):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    mask_token_pair_label responding the label of the pair where this token is from
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []
    mask_token_pair_label = []

    if pretraining_task is 'seq2seq':
        assert sent_b_starts is not None, \
            "[FATAL] For seq2seq language model loss," \
            " sent_b_starts should not be None"
        for sent_index, sent in enumerate(batch_tokens):
            sent_b_index = sent_b_starts[sent_index]
            mask_label_cur = sent[sent_b_index + 1:]  # from the second target word until the end
            mask_pos_cur = [
                sent_index * max_len + sent_b_index + i
                for i in range(len(sent[sent_b_index:]))
            ]  # from target word index to the end

            if del_pos:  # remove the index of [cls_id] between target spans
                map(lambda i: mask_label_cur.pop(i), del_pos[sent_index])
                map(lambda i: mask_pos_cur.pop(i - 1), del_pos[sent_index])

            # the output labels for the masked positions
            mask_label.extend(mask_label_cur)
            # traditional generation: the masked index of context stream,
            # which use the current token to predict the next token
            mask_pos.extend(mask_pos_cur[:-1])
            mask_token_pair_label.extend([pair_labels[sent_index]] * len(mask_label_cur))

        assert len(mask_pos) == len(mask_label) == len(mask_token_pair_label), "the mask len is invalid"
        mask_label = np.array(mask_label).astype('int64').reshape([-1, 1])
        mask_pos = np.array(mask_pos).astype('int64').reshape([-1, 1])
        mask_token_pair_label = np.array(mask_token_pair_label).astype("float32").reshape([-1, 1])
        return batch_tokens, mask_label, mask_pos, mask_token_pair_label

    assert geometric_p > 0, "geometric_p must be larger than 0."
    span_lens = list(range(span_lower, span_upper + 1))
    len_distrib = [geometric_p * (1 - geometric_p) ** (i - span_lower) for i in span_lens]
    len_distrib = [x / (sum(len_distrib)) for x in len_distrib]

    pre_sent_len = 0
    prob_index = 0
    new_batch_tokens = []
    for sent_index, sent in enumerate(batch_tokens):
        sent = copy.deepcopy(sent)
        prob_index += pre_sent_len

        if batch_can_mask is not None and batch_can_mask[sent_index]:
            sentence, sent_mask_label, sent_mask_pos = mask_span(sent, seg_labels[sent_index], txt_mask_ratio,
                                                                 vocab_size=vocab_size, CLS=CLS, SEP=SEP, MASK=MASK,
                                                                 span_lens=span_lens, len_distrib=len_distrib)
            batch_tokens[sent_index] = sentence
            mask_label.extend(sent_mask_label)
            for pos_index in range(len(sent_mask_pos)):
                mask_pos.append(sent_mask_pos[pos_index] + sent_index * max_len)
            mask_token_pair_label.extend([pair_labels[sent_index]] * len(sent_mask_label))

        pre_sent_len = len(sent)
        new_batch_tokens.append(sent)

    assert (prob_index + pre_sent_len) == total_token_num, 'the total_token_num is incorrect'
    while len(mask_pos) < 2:
        rand_index = np.random.randint(low=1, high=total_token_num)
        pre_seq_len = 0
        for sent_index, sent in enumerate(batch_tokens):
            if rand_index < pre_seq_len + len(sent):
                mask_pos.append(rand_index - pre_seq_len + sent_index * max_len)
                mask_label.append(sent[rand_index - pre_seq_len])
                mask_token_pair_label.append(0)
                break
            pre_seq_len += len(sent)

    assert len(mask_pos) == len(mask_label) == len(mask_token_pair_label), "the mask len is invalid"
    mask_label = np.array(mask_label).astype('int64').reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype('int64').reshape([-1, 1])
    mask_token_pair_label = np.array(mask_token_pair_label).astype("float32").reshape([-1, 1])
    return new_batch_tokens, mask_label, mask_pos, mask_token_pair_label


def mask_span(sentence,
              seg_label,
              txt_mask_ratio=0.15,
              vocab_size=0,
              CLS=0,
              SEP=2,
              MASK=50264,
              span_lens=None,
              len_distrib=None):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    mask_token_pair_label responding the label of the pair where this token is from
    """
    sent_length = len(sentence)
    mask_num = math.ceil(sent_length * txt_mask_ratio)
    mask_pos = set()
    spans = []
    max_iteration = 0
    while len(mask_pos) < mask_num and sent_length > 2:
        max_iteration += 1
        if max_iteration > 100 * mask_num:
            break

        span_len = np.random.choice(span_lens, p=len_distrib)
        anchor = np.random.choice(sent_length)
        if anchor in mask_pos or seg_label[anchor] == -1:
            continue

        # find word start, end
        left1, right1 = get_word_start(anchor, seg_label, sent_length), \
                        get_word_end(anchor, seg_label, sent_length)
        spans.append([left1, left1])

        for i in range(left1, right1):
            if len(mask_pos) >= mask_num:
                break
            mask_pos.add(i)
            spans[-1][-1] = i

        num_words = 1
        right2 = right1
        while num_words < span_len and right2 < len(sentence) and len(mask_pos) < mask_num:
            # complete current word
            left2 = right2
            right2 = get_word_end(right2, seg_label, sent_length)
            num_words += 1
            for i in range(left2, right2):
                if len(mask_pos) >= mask_num:
                    break
                mask_pos.add(i)
                spans[-1][-1] = i

    sentence, mask_label, ordered_mask_pos = span_masking(sentence, spans, vocab_size,
                                                          mask_id=MASK, mask_pos=mask_pos)
    return sentence, mask_label, ordered_mask_pos


def span_masking(sentence, spans, vocab_size, mask_id=50264, mask_pos=None):
    sent_length = len(sentence)
    replace_ids = np.random.randint(4, high=vocab_size, size=sent_length)  # exclude CLS, PAD, SEP and MASK
    spans = merge_intervals(spans)
    assert len(mask_pos) == sum([e - s + 1 for s, e in spans])
    mask_label = []
    ordered_mask_pos = []
    for start, end in spans:
        for i in range(start, end + 1):
            rand = np.random.random()
            assert i in mask_pos
            if rand < 0.8:
                ordered_mask_pos.append(i)
                mask_label.append(sentence[i])
                sentence[i] = mask_id
            elif rand < 0.9:
                ordered_mask_pos.append(i)
                mask_label.append(sentence[i])
                # sample random token according to input distribution
                sentence[i] = replace_ids[i]
            else:
                ordered_mask_pos.append(i)
                mask_label.append(sentence[i])
    assert len(mask_label) == len(ordered_mask_pos), 'the length of mask_label and mask_pos are not equal!'
    return sentence, mask_label, ordered_mask_pos


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] + 1 < interval[0]:
            merged.append(interval)
        else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def get_word_start(anchor, seg_label, sent_length):
    if seg_label[anchor] == 0:
        return anchor
    else:
        start = anchor
        while start >= 0 and seg_label[start] == 1:
            start -= 1

        return max(start, 0)


def get_word_end(anchor, seg_label, sent_length):
    if seg_label[anchor] == 0:
        if anchor + 1 >= sent_length:
            return anchor + 1
        if seg_label[anchor + 1] == 0 or seg_label[anchor + 1] == -1:
            return anchor + 1
        else:
            end = anchor + 1
            while end < sent_length and seg_label[end] == 1:
                end += 1

            return end
    else:
        end = anchor
        while end < sent_length and seg_label[end] == 1:
            end += 1

        return end


def _except(sent_index, batch_del_pos, batch_src_ids, batch_pos_ids,
            seg_labels, batch_sent_ids, sent_b_starts, mask_id, cls_id, sep_id):
    """making all src_ids as target
    src_ids: [CLS] [SEP] [CLS] t1 t2 ... tn
    pos_ids: [0, 1, 0, 1, ..., n-1]
    seg_labels: [-1, -1, -1, 0, 0, ...,0]
    sent_ids: [0, 0, 0, 0, ..., 0]
    sent_b_starts: [2]
    """
    batch_del_pos.append([])
    batch_src_ids[sent_index] = [cls_id, sep_id] + batch_src_ids[sent_index][:510]
    batch_pos_ids[sent_index] = [0, 1] + list(range(0, len(batch_src_ids[sent_index]) - 2))
    seg_labels[sent_index] = [-1, -1] + seg_labels[sent_index][:510]
    batch_sent_ids[sent_index] = [0] * len(seg_labels[sent_index])
    sent_b_starts.append(2)


def prepare_batch_data(insts,
                       total_token_num,
                       txt_mask_ratio=0.15,
                       vl_mask_ratio=0.15,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       patch_seq_len=196,
                       patch_emb_size=768):
    # text part
    batch_src_ids = [inst[1] for inst in insts]
    batch_sent_ids = [inst[2] for inst in insts]
    batch_pos_ids = [inst[3] for inst in insts]
    labels = [inst[4] for inst in insts]
    labels = np.array(labels).astype("int64").reshape([-1, 1])
    pair_labels = labels[:, 0]
    seg_labels = [inst[5] for inst in insts]
    batch_can_mask = [inst[6] for inst in insts]

    sent_b_starts = []
    batch_del_pos = []

    # pretraining_task = ['mlm', 'seq2seq']
    # pretraining_prob = [0.5, 0.5]
    # # choose which pretraining task to use
    # sample_task = np.random.choice(pretraining_task, p=pretraining_prob)

    sample_task = "mlm"
    if sample_task is 'seq2seq':
        for sent_index, sent in enumerate(batch_src_ids):
            sample_info = [
                {"en_slice_min": 3, "en_slice_max": 12, "de_slice_min": 1, "de_slice_max": 4},
                {"en_slice_min": 12, "en_slice_max": 96, "de_slice_min": 4, "de_slice_max": 32},
            ]
            sample_prob = [0.4, 0.6]

            loop_max = 1000
            loop_stop = False
            pos_start = 2
            if len(sent) >= sample_info[0]["de_slice_min"] + sample_info[0]["en_slice_min"] + 2:
                loop_cnt = 0
                split_idx = [1]
                start_choice = np.random.choice(2)  # mask encoder part or decoder part
                while len(split_idx) < 2 or (len(split_idx) == 2 and split_idx[-1] == len(sent)):
                    if loop_cnt >= loop_max:
                        loop_stop = True
                        break
                    loop_cnt += 1
                    mass_len, seg_id = 0, -1
                    split_idx = [1]

                    # choose which sample_info to use
                    sample = np.random.choice(sample_info, p=sample_prob)

                    # segment length to be masked
                    if start_choice == 0:
                        seg_len = np.random.choice(range(sample['en_slice_min'], sample['en_slice_max'] + 1))
                    else:
                        seg_len = np.random.choice(range(sample['de_slice_min'], sample['de_slice_max'] + 1))

                    # don't include the first [CLS] token and the last [SEP] token
                    while len(sent) - split_idx[-1] > seg_len:
                        split_idx.append(seg_len + split_idx[-1])
                        seg_id += 1
                        sample = np.random.choice(sample_info, p=sample_prob)
                        if seg_id % 2 == start_choice:
                            seg_len = np.random.choice(range(sample['de_slice_min'], sample['de_slice_max'] + 1))
                        else:
                            seg_len = np.random.choice(range(sample['en_slice_min'], sample['en_slice_max'] + 1))

                # include the last [SEP] token
                if split_idx[-1] != len(sent):
                    split_idx.append(len(sent))

                if loop_stop:
                    _except(sent_index, batch_del_pos, batch_src_ids, batch_pos_ids,
                            seg_labels, batch_sent_ids, sent_b_starts, mask_id, cls_id, sep_id)
                    print("loop stop")
                    continue

                en_src_ids, de_src_ids = [cls_id], []
                en_pos_ids, de_pos_ids = [pos_start], []
                en_seg_labels, de_seg_labels = [-1], []

                """en_src_ids: [CLS] span1 span2 span3 ... [SEP]
                   de_src_ids: [CLS] span1 [CLS] span2 [CLS] span3 ...
                """
                for i in range(1, len(split_idx)):
                    if i % 2 != start_choice:
                        en_src_ids.extend(batch_src_ids[sent_index][split_idx[i - 1]:split_idx[i]])
                        en_pos_ids.extend(batch_pos_ids[sent_index][split_idx[i - 1]:split_idx[i]])
                        en_seg_labels.extend(seg_labels[sent_index][split_idx[i - 1]:split_idx[i]])
                    else:
                        de_src_ids.append([cls_id] + batch_src_ids[sent_index][split_idx[i - 1]:split_idx[i]])
                        de_pos_ids.append(batch_pos_ids[sent_index][split_idx[i - 1] - 1:split_idx[i]])
                        de_seg_labels.append([-1] + seg_labels[sent_index][split_idx[i - 1]:split_idx[i]])

                # include the last [SEP] token
                en_src_ids.append(sep_id)
                en_pos_ids.append(batch_pos_ids[sent_index][-1])
                en_seg_labels.append(-1)
                de_src_ids, de_pos_ids, de_seg_labels = sum(de_src_ids, []), sum(de_pos_ids, []), sum(de_seg_labels, [])

                del_pos = []
                # remove the index of [cls_id] from mask_pos except for the first one
                for i, src_id in enumerate(de_src_ids):
                    if i and src_id == 1:
                        del_pos.append(i - len(de_src_ids))

                batch_del_pos.append(del_pos)
                batch_src_ids[sent_index] = en_src_ids + de_src_ids
                batch_pos_ids[sent_index] = en_pos_ids + de_pos_ids
                batch_sent_ids[sent_index] = [0] * (len(en_src_ids) + len(de_src_ids))
                seg_labels[sent_index] = en_seg_labels + de_seg_labels
                sent_b_starts.append(len(en_src_ids))
            else:
                _except(sent_index, batch_del_pos, batch_src_ids, batch_pos_ids,
                        seg_labels, batch_sent_ids, sent_b_starts, mask_id, cls_id, sep_id)

    # First step: do mask without padding
    assert mask_id >= 0, "[FATAL] mask_id must >= 0"
    out, mask_label, mask_pos, mask_token_pair_label = mask(
        batch_src_ids,
        seg_labels,
        total_token_num,
        txt_mask_ratio=txt_mask_ratio,
        pair_labels=pair_labels,
        vocab_size=voc_size,
        pretraining_task=sample_task,
        CLS=cls_id,
        SEP=sep_id,
        MASK=mask_id,
        sent_b_starts=sent_b_starts,
        del_pos=batch_del_pos,
        batch_can_mask=batch_can_mask)

    # Second step: padding
    """self_input_mask: all tokens can attend src part, each tgt word only attend its own 
    and previous tgt words, used for word-by-word generation"""
    src_id, text_mask = pad_batch_data(
        out,
        pretraining_task=sample_task,
        sent_b_starts=sent_b_starts,
        pad_idx=pad_id,
        return_input_mask=True)

    pos_id = pad_batch_data(batch_pos_ids, pad_idx=pad_id)
    sent_id = pad_batch_data(batch_sent_ids, pad_idx=pad_id)

    # visual part
    batch_image_pixel = [inst[8] for inst in insts]
    batch_image_pixel_pos = [inst[9] for inst in insts]
    batch_with_image = [inst[11] for inst in insts]
    batch_with_text = [inst[12] for inst in insts]
    batch_is_multimodal = [inst[13] for inst in insts]
    total_img_token_num = patch_seq_len * len(batch_image_pixel)  # don't include the global image token

    with_image = np.array(batch_with_image, dtype='float32')
    with_image = with_image.reshape((-1, 1))
    with_text = np.array(batch_with_text, dtype='float32')
    with_text = with_text.reshape((-1, 1))
    is_multimodal = np.array(batch_is_multimodal, dtype='float32')
    is_multimodal = is_multimodal.reshape((-1, 1))

    # image pixels, include the global image token
    image_mask = np.ones(shape=[len(insts), 1, patch_seq_len + 1], dtype="float32")
    image_pixel_input = np.array(batch_image_pixel, dtype='float32')

    text_batch = [src_id, pos_id, sent_id, text_mask, mask_label, mask_pos, mask_token_pair_label]
    image_batch = [image_pixel_input, image_mask, with_image, with_text, is_multimodal, labels]
    return_list = text_batch + image_batch

    return return_list


def pad_batch_data(insts,
                   pretraining_task='seq2seq',
                   pad_idx=1,
                   sent_b_starts=None,
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

    inst_data = np.array(
        [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype('int64').reshape([-1, max_len, 1])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype('int64').reshape([-1, max_len, 1])]

    if return_input_mask:
        if pretraining_task is 'seq2seq':
            assert sent_b_starts is not None, \
                "[FATAL] For seq2seq lanugae model loss," \
                " sent_b_starts should not be None"
            # This is used to avoid attention on paddings and subsequent words.
            input_mask_data = np.zeros((inst_data.shape[0], max_len, max_len))
            for index, mask_data in enumerate(input_mask_data):
                start = sent_b_starts[index]
                end = len(insts[index])
                mask_data[:end, :start] = 1.0
                # Generate the lower triangular matrix using the slice of matrix
                b = np.tril(np.ones([end - start, end - start]), 0)
                mask_data[start:end, start:end] = b
            input_mask_data = np.array(input_mask_data).reshape([-1, max_len, max_len])
        else:
            # This is used to avoid attention on paddings.
            input_mask_data = np.array([[1] * len(inst) + [0] *
                                        (max_len - len(inst)) for inst in insts])
            input_mask_data = np.expand_dims(input_mask_data, axis=1)
            # input_mask_data = np.matmul(input_mask_data, np.transpose(input_mask_data, (0, 2, 1)))
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


def gen_seq2seq_mask(insts, batch_src_lens=None, sent_b_starts=None):
    """
    generate input mask for seq2seq
    """
    max_len = max(len(inst) for inst in insts)
    input_mask_data = np.zeros((len(insts), max_len, max_len))
    for index, mask_data in enumerate(input_mask_data):
        start = sent_b_starts[index]
        end = len(insts[index])
        if batch_src_lens:
            src_len = batch_src_lens[index]
            mask_data[:end, :src_len] = 1.0
            mask_data[src_len:start, :] = 0.0
        else:
            mask_data[:end, :start] = 1.0
        # Generate the lower triangular matrix using the slice of matrix
        b = np.tril(np.ones([end - start, end - start]), 0)
        mask_data[start:end, start:end] = b
    input_mask_data = np.array(input_mask_data, dtype='float32').reshape([-1, max_len, max_len])
    return input_mask_data


if __name__ == "__main__":
    pass
