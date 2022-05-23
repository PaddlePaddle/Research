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
File: retrieval_grounded_reader.py
Author: liwei(liwei85@baidu.com)
Date: 2021-08-23 14:26
Desc: data reader for image-text retrieval tasks
"""

import os
import pickle
import base64
import codecs
import numpy as np
from collections import namedtuple
from reader.unimo_grounded_batching import pad_batch_data
from utils.image_utils import process_image_with_multi_proc, process_image
from functools import partial


class RetrievalTrainReader(object):
    """RetrievalTrainReader"""

    def __init__(self, tokenizer, args, image_caption, image_size, resolution):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.max_seq_len = args.max_seq_len

        self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
        self.trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))

        self.image_size = image_size
        assert resolution in {14, 16}
        self.patch_seq_len = self.image_size * self.image_size // (resolution * resolution)
        self.patch_emb_size = resolution * resolution * 3

        self.current_example = 0
        self.current_epoch = 0
        self._load_caption_image_dict(image_caption)

        if args.samples_num == 20:
            self._negative_schema = ['ei'] * 10 + ['ec'] * 10
            self.outs = len(self._negative_schema) + 1
        else:
            raise ValueError('dont support')

    def _load_caption_image_dict(self, image_caption, proc_num=8):
        '''parse dataset_flickr30k.json which is made by karpathy'''
        self._caption_ids_dict = {}
        self._image_sent_map = {}

        with codecs.open(image_caption, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(";")
                token_ids, sent_ids, pos_ids, image_id, sent_id = line
                token_ids = [int(token) for token in token_ids.split(" ")]
                sent_ids = [int(token) for token in sent_ids.split(" ")]
                pos_ids = [int(token) for token in pos_ids.split(" ")]
                if len(token_ids) > self.max_seq_len:
                    token_ids = [token_ids[0]] + token_ids[1:self.max_seq_len - 1] + [token_ids[-1]]
                    sent_ids = sent_ids[:self.max_seq_len]
                    pos_ids = pos_ids[:self.max_seq_len]
                assert len(token_ids) <= self.max_seq_len, \
                    "token length must be less than max_seq_len"
                assert len(token_ids) == len(sent_ids) == len(pos_ids), \
                    "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"

                self._caption_ids_dict[int(sent_id)] = \
                    [token_ids, sent_ids, pos_ids, int(image_id)]
                self._image_sent_map.setdefault(int(image_id), [])
                self._image_sent_map[int(image_id)].append(int(sent_id))

        self._train_caption_ids = list(self._caption_ids_dict.keys())
        self._train_image_list = list(self._image_sent_map.keys())

        image_base_dir = './data/Flickr30k/flickr30k-images'
        image_id_paths = [(image_id, os.path.join(image_base_dir, str(image_id) + '.jpg'))
                          for image_id in self._train_image_list]
        cur_process_image = partial(process_image,
                                    target_shape_h=self.image_size,
                                    target_shape_w=self.image_size)
        image_id_pixels_list = process_image_with_multi_proc(image_id_paths, proc_num, cur_process_image)
        self._image_id_pixels_dict = dict(image_id_pixels_list)

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _prepare_batch_data(self, insts):
        """generate batch and pad"""
        batch_src_ids = [inst["token_ids"][out] for inst in insts for out in range(self.outs)]
        batch_sent_ids = [inst["sent_ids"][out] for inst in insts for out in range(self.outs)]
        batch_pos_ids = [inst["pos_ids"][out] for inst in insts for out in range(self.outs)]
        batch_image_pixel = [inst["image_pixel_inputs"][out] for inst in insts for out in range(self.outs)]

        batch_size = int(len(batch_src_ids) / self.outs)
        label = np.array([[0]] * batch_size, dtype="int64")
        ids = np.array([[0, 0]] * batch_size, dtype="int64")

        padded_token_ids, token_mask = pad_batch_data(
            batch_src_ids, pretraining_task='nlu', pad_idx=self.pad_id, return_input_mask=True)
        padded_sent_ids = pad_batch_data(
            batch_sent_ids, pretraining_task='nlu', pad_idx=self.pad_id)
        padded_pos_ids = pad_batch_data(
            batch_pos_ids, pretraining_task='nlu', pad_idx=self.pad_id)

        # image pixels, include the global image token
        image_mask = np.ones(shape=[len(batch_image_pixel), 1, self.patch_seq_len + 1], dtype="float32")
        image_pixel_input = np.array(batch_image_pixel, dtype='float32')

        return_list = [
            padded_token_ids, padded_pos_ids, padded_sent_ids, token_mask,
            image_pixel_input, image_mask, label, ids
        ]
        return return_list

    def get_num_examples(self):
        """get_num_examples"""
        cap_len = len(self._train_caption_ids)
        img_len = len(self._train_image_list)
        total_samples = cap_len
        return total_samples, cap_len, img_len

    def process_vl(self, sent_id):
        """trans the orgin tokens to the wanted tokens"""
        token_ids, sent_ids, pos_ids, image_id = self._caption_ids_dict[sent_id]
        image_pixel_input = self._image_id_pixels_dict[image_id]

        images = [image_pixel_input]
        captions = [[token_ids, sent_ids, pos_ids]]

        for item in self._negative_schema:
            if item[0] == "e":
                while True:
                    image_id_neg = self.neg_rng.choice(self._train_image_list)
                    if image_id_neg != image_id:
                        break
            else:
                print("error negative schema")
                exit()

            if item[1] == "i":
                image_pixel_input_neg = self._image_id_pixels_dict[image_id_neg]
                captions.append(self._caption_ids_dict[sent_id][:-1])
                images.append(image_pixel_input_neg)
            elif item[1] == "c":
                sent_id_neg = self.neg_rng.choice(self._image_sent_map[image_id_neg])
                captions.append(self._caption_ids_dict[sent_id_neg][:-1])
                images.append(image_pixel_input)
            else:
                print("error negative schema")
                exit()

        token_ids_list, sent_ids_list, pos_ids_list = zip(*captions)
        image_pixel_input_list = images

        sample_json = {
            "token_ids": token_ids_list,
            "sent_ids": sent_ids_list,
            "pos_ids": pos_ids_list,
            "image_pixel_inputs": image_pixel_input_list
        }
        return sample_json

    def read_caption_id(self):
        """read_caption_id"""
        self.global_rng.shuffle(self._train_caption_ids)
        for index, item in enumerate(self._train_caption_ids):
            if index % self.trainers_num != self.trainer_id:
                continue
            yield self.process_vl(item)

    def shuffle_samples(self, sample_generator, buffer=128):
        """shuffle_samples"""
        samples = []
        try:
            while True:
                while len(samples) < buffer:
                    sample = next(sample_generator)
                    samples.append(sample)
                for sample in samples:
                    yield sample
                samples = []
        except StopIteration:
            if len(samples) == 0:
                yield None
            else:
                for sample in samples:
                    yield sample

    def data_generator(self):
        """data_generator"""

        def wrapper():
            """wrapper"""
            for epoch_index in range(self.epoch):
                self.global_rng = np.random.RandomState(epoch_index)
                self.neg_rng = np.random.RandomState(epoch_index)
                self.current_epoch = epoch_index
                batch_records = []
                self.current_example = 0
                for sample in self.shuffle_samples(self.read_caption_id()):
                    self.current_example = self.current_example + 1
                    if len(batch_records) < self.batch_size:
                        batch_records.append(sample)
                    if len(batch_records) == self.batch_size:
                        yield self._prepare_batch_data(batch_records)
                        batch_records = []
                if batch_records:
                    yield self._prepare_batch_data(batch_records)

        return wrapper


class RetrievalTestReader(object):
    """RetrievalTrainReader"""

    def __init__(self, tokenizer, args, image_caption, image_size, resolution):
        self.batch_size = args.test_batch_size
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.max_seq_len = args.max_seq_len
        self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
        self.trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        self.current_example = 0

        self.image_size = image_size
        assert resolution in {14, 16}
        self.patch_seq_len = self.image_size * self.image_size // (resolution * resolution)
        self.patch_emb_size = resolution * resolution * 3

        self._load_caption_image_dict(image_caption)

    def _load_caption_image_dict(self, image_caption, proc_num=8):
        '''parse dataset_flickr30k.json which is made by karpathy'''
        self._caption_ids_dict = {}
        self._image_sent_map = {}

        with codecs.open(image_caption, 'r', encoding='utf-8') as f:
            cnt = 0
            for line in f:
                line = line.strip().split(";")
                token_ids, sent_ids, pos_ids, image_id, sent_id = line
                token_ids = [int(token) for token in token_ids.split(" ")]
                sent_ids = [int(token) for token in sent_ids.split(" ")]
                pos_ids = [int(token) for token in pos_ids.split(" ")]
                if len(token_ids) > self.max_seq_len:
                    token_ids = [token_ids[0]] + token_ids[1:self.max_seq_len - 1] + [token_ids[-1]]
                    sent_ids = sent_ids[:self.max_seq_len]
                    pos_ids = pos_ids[:self.max_seq_len]
                assert len(token_ids) <= self.max_seq_len, \
                    "token length must be less than max_seq_len"
                assert len(token_ids) == len(sent_ids) == len(pos_ids), \
                    "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids)"

                self._caption_ids_dict[int(sent_id)] = \
                    [token_ids, sent_ids, pos_ids, int(image_id)]
                self._image_sent_map.setdefault(int(image_id), [])
                self._image_sent_map[int(image_id)].append(int(sent_id))

        self._train_caption_ids = list(self._caption_ids_dict.keys())
        self._train_image_list = list(self._image_sent_map.keys())

        image_base_dir = './data/Flickr30k/flickr30k-images'
        image_id_paths = [(image_id, os.path.join(image_base_dir, str(image_id) + '.jpg'))
                          for image_id in self._train_image_list]
        cur_process_image = partial(process_image,
                                    target_shape_h=self.image_size,
                                    target_shape_w=self.image_size)
        image_id_pixels_list = process_image_with_multi_proc(image_id_paths, proc_num, cur_process_image)
        self._image_id_pixels_dict = dict(image_id_pixels_list)

    def _prepare_batch_data(self, insts):
        """generate batch and pad"""
        batch_src_ids = [inst["token_ids"] for inst in insts]
        batch_sent_ids = [inst["sent_ids"] for inst in insts]
        batch_pos_ids = [inst["pos_ids"] for inst in insts]
        batch_image_pixel = [inst["image_pixel_input"] for inst in insts]
        batch_ids = [inst["cur_ids"] for inst in insts]
        batch_labels = [[0]] * len(insts)

        padded_token_ids, token_mask = pad_batch_data(
            batch_src_ids, pretraining_task='nlu', pad_idx=self.pad_id, return_input_mask=True)
        padded_sent_ids = pad_batch_data(
            batch_sent_ids, pretraining_task='nlu', pad_idx=self.pad_id)
        padded_pos_ids = pad_batch_data(
            batch_pos_ids, pretraining_task='nlu', pad_idx=self.pad_id)

        # image pixels, include the global image token
        image_mask = np.ones(shape=[len(batch_image_pixel), 1, self.patch_seq_len + 1], dtype="float32")
        image_pixel_input = np.array(batch_image_pixel, dtype='float32')

        ids = np.array(batch_ids, dtype="int64")
        label = np.array(batch_labels, dtype="int64")

        return_list = [
            padded_token_ids, padded_pos_ids, padded_sent_ids, token_mask,
            image_pixel_input, image_mask, label, ids
        ]
        return return_list

    def get_num_examples(self):
        """get_num_examples"""
        cap_len = len(self._train_caption_ids)
        img_len = len(self._train_image_list)
        total_samples = cap_len
        return total_samples, cap_len, img_len

    def read_caption_id(self):
        """read_caption_id"""
        for sent_id in self._train_caption_ids:
            token_ids, sent_ids, pos_ids, _ = self._caption_ids_dict[sent_id]

            for cur_img_id in self._train_image_list:
                cur_image_pixel_input = self._image_id_pixels_dict[cur_img_id]
                sample_json = {
                    "token_ids": token_ids,
                    "sent_ids": sent_ids,
                    "pos_ids": pos_ids,
                    "image_pixel_input": cur_image_pixel_input,
                    "cur_ids": [cur_img_id, sent_id]
                }
                yield sample_json

    def shuffle_samples(self, sample_generator, buffer=128):
        """shuffle_samples"""
        samples = []
        try:
            while True:
                while len(samples) < buffer:
                    sample = next(sample_generator)
                    samples.append(sample)
                for sample in samples:
                    yield sample
                samples = []
        except StopIteration:
            if len(samples) == 0:
                yield None
            else:
                for sample in samples:
                    yield sample

    def data_generator(self):
        """data_generator"""

        def wrapper():
            """"wrapper"""

            def batch_reader():
                """batch_reader"""
                batch_records = []
                self.current_example = 0
                for sample in self.shuffle_samples(self.read_caption_id()):
                    self.current_example = self.current_example + 1
                    if len(batch_records) < self.batch_size:
                        batch_records.append(sample)
                    if len(batch_records) == self.batch_size:
                        yield self._prepare_batch_data(batch_records)
                        batch_records = []
                if batch_records:
                    yield self._prepare_batch_data(batch_records)

            all_dev_batches = []
            for batch_data in batch_reader():
                if len(all_dev_batches) < self.trainers_num:
                    all_dev_batches.append(batch_data)
                if len(all_dev_batches) == self.trainers_num:
                    yield all_dev_batches[self.trainer_id]
                    all_dev_batches = []
            if self.trainer_id < len(all_dev_batches):
                yield all_dev_batches[self.trainer_id]

        return wrapper


if __name__ == '__main__':
    pass
