# coding: utf-8
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
"""task reader
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import os
import json
import random
import logging
import numpy as np
import six
from io import open
from collections import namedtuple

import tokenization
from batching import pad_batch_data

log = logging.getLogger(__name__)

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def csv_reader(fd, delimiter='\t'):
    """csv_reader"""

    def gen():
        """gen"""
        for i in fd:
            yield i.rstrip('\n').split(delimiter)

    return gen()


class BaseReader(object):
    """BaseReader
    """

    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id

        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        if label_map_config:
            with open(label_map_config, encoding='utf8') as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f)
            headers = next(reader)
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                example = Example(*line)
                examples.append(example)
            return examples

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        text_a = tokenization.convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None

        has_text_b = False
        if isinstance(example, dict):
            has_text_b = "text_b" in example.keys()
        else:
            has_text_b = "text_b" in example._fields

        if has_text_b:
            text_b = tokenization.convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT/ERNIE is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))

        if self.is_inference:
            Record = namedtuple(
                'Record', ['token_ids', 'text_type_ids', 'position_ids'])
            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids)
        else:
            if self.label_map:
                label_id = self.label_map[example.label]
            else:
                label_id = example.label

            Record = namedtuple('Record', [
                'token_ids', 'text_type_ids', 'position_ids', 'label_id', 'qid'
            ])

            qid = None
            if "qid" in example._fields:
                qid = example.qid

            record = Record(
                token_ids=token_ids,
                text_type_ids=text_type_ids,
                position_ids=position_ids,
                label_id=label_id,
                qid=qid)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_reords, max_len = [], 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        """func"""
        examples = self._read_tsv(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        """func"""
        examples = self._read_tsv(input_file)

        def wrapper():
            """func"""
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        return wrapper


class TriggerSequenceLabelReader(BaseReader):
    """TriggerSequenceLabelReader
    """

    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 labels_map=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id

        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0

        self.label_map = labels_map

    def _process_examples_by_json(self, input_data):
        """_examples_by_json"""

        def process_sent_ori_2_new(sent, start, end):
            """process_sent_ori_2_new"""
            words = list(sent)
            sent_ori_2_new_index = {}
            new_words = []
            new_start, new_end = -1, -1
            for i, w in enumerate(words):
                if i == start:
                    new_start = len(new_words)
                if i == end:
                    new_end = len(new_words)
                if len(w.strip()) == 0:
                    sent_ori_2_new_index[i] = -1
                    if i == end:
                        new_end -= 1
                    if i == start:
                        start += 1
                else:
                    sent_ori_2_new_index[i] = len(new_words)
                    new_words.append(w)
            if new_end == len(new_words):
                new_end = len(new_words) - 1

            return [words, new_words, sent_ori_2_new_index, new_start, new_end]

        examples = []
        k = 0
        Example = namedtuple('Example', [
            "id", "text_a", "label", "ori_text", "ori_2_new_index", "sentence"
        ])
        for data in input_data:
            event_id = data["event_id"]
            sentence = data["text"]
            trigger_start = data["trigger_start_index"]
            trigger_text = data["trigger"]
            trigger_end = trigger_start + len(trigger_text) - 1
            event_type = data["event_type"]
            (sent_words, new_sent_words, ori_2_new_sent_index,
             new_trigger_start, new_trigger_end) = process_sent_ori_2_new(
                 sentence.lower(), trigger_start, trigger_end)
            new_sent_labels = [u"O"] * len(new_sent_words)
            for i in range(new_trigger_start, new_trigger_end + 1):
                if i == new_trigger_start:
                    new_sent_labels[i] = u"B-{}".format(event_type)
                else:
                    new_sent_labels[i] = u"I-{}".format(event_type)
            example = Example(
                id=event_id,
                text_a=u" ".join(new_sent_words),
                label=u" ".join(new_sent_labels),
                ori_text=sent_words,
                ori_2_new_index=ori_2_new_sent_index,
                sentence=sentence)

            if k > 0:
                print(u"example {} : {}".format(
                    k, json.dumps(
                        example._asdict(), ensure_ascii=False)))
            k -= 1
            examples.append(example)
        return examples

    def _read_json_file(self, input_file):
        """_read_json_file"""
        input_data = []
        with open(input_file, "r", encoding='utf8') as f:
            for line in f:
                d_json = json.loads(line.strip())
                input_data.append(d_json)
        examples = self._process_examples_by_json(input_data)
        return examples

    def get_examples_by_file(self, input_file):
        """get_examples_by_file"""
        return self._read_json_file(input_file)

    def _pad_batch_records(self, batch_records):
        """_pad_batch_records"""
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [
            record.text_type_ids for record in batch_records
        ]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(
            batch_label_ids, pad_idx=len(self.label_map) - 1)
        padded_task_ids = np.ones_like(
            padded_token_ids, dtype="int64") * self.task_id

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, padded_label_ids, batch_seq_lens
        ]
        return return_list

    def _reseg_token_label(self, tokens, labels, tokenizer):
        """_reseg_token_label"""
        assert len(tokens) == len(labels)
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            if len(sub_token) == 1:
                ret_labels.append(label)
                continue

            if label == "O" or label.startswith("I-"):
                ret_labels.extend([label] * len(sub_token))
            elif label.startswith("B-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([label] + [i_label] * (len(sub_token) - 1))

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """_convert_example_to_record"""
        tokens = tokenization.whitespace_tokenize(example.text_a)
        labels = tokenization.whitespace_tokenize(example.label)
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        text_type_ids = [0] * len(token_ids)
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id
                     ] + [self.label_map[label]
                          for label in labels] + [no_entity_id]

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_ids'])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_ids=label_ids)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        k = 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer)
            if k > 0:
                print(u"feature {} : {}".format(
                    k, json.dumps(
                        record._asdict(), ensure_ascii=False)))
            k -= 1
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        """get_num_examples"""
        examples = self._read_json_file(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        """data_generator"""
        examples = self._read_json_file(input_file)

        def wrapper():
            """wrapper"""
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        return wrapper


class RoleSequenceLabelReader(BaseReader):
    """RoleSequenceLabelReader
    """

    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 labels_map=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 is_inference=False,
                 random_seed=None,
                 tokenizer="FullTokenizer",
                 is_classify=True,
                 is_regression=False,
                 for_cn=True,
                 task_id=0):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.in_tokens = in_tokens
        self.is_inference = is_inference
        self.for_cn = for_cn
        self.task_id = task_id

        np.random.seed(random_seed)

        self.is_classify = is_classify
        self.is_regression = is_regression
        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = 0
        self.label_map = labels_map

    def _process_examples_by_json(self, input_data):
        """_examples_by_json"""

        def process_sent_ori_2_new(sent, roles_list):
            """process_sent_ori_2_new"""
            words = list(sent)
            sent_ori_2_new_index = {}
            new_words = []
            new_start, new_end = -1, -1
            new_roles_list = {}
            for role_type, role in roles_list.items():
                new_roles_list[role_type] = {
                    "role_type": role_type,
                    "start": -1,
                    "end": -1
                }

            for i, w in enumerate(words):
                for role_type, role in roles_list.items():
                    if i == role["start"]:
                        new_roles_list[role_type]["start"] = len(new_words)
                    if i == role["end"]:
                        new_roles_list[role_type]["end"] = len(new_words)

                if len(w.strip()) == 0:
                    sent_ori_2_new_index[i] = -1
                    for role_type, role in roles_list.items():
                        if i == role["start"]:
                            new_roles_list[role_type]["start"] += 1
                        if i == role["end"]:
                            new_roles_list[role_type]["end"] -= 1
                else:
                    sent_ori_2_new_index[i] = len(new_words)
                    new_words.append(w)
            for role_type, role in new_roles_list.items():
                if role["start"] > -1:
                    role["text"] = u"".join(new_words[role["start"]:role["end"]
                                                      + 1])
                if role["end"] == len(new_words):
                    role["end"] = len(new_words) - 1

            return [words, new_words, sent_ori_2_new_index, new_roles_list]

        examples = []
        k = 0
        Example = namedtuple('Example', [
            "id", "text_a", "label", "ori_text", "ori_2_new_index", "roles",
            "sentence"
        ])
        for data in input_data:
            event_id = data["event_id"]
            sentence = data["text"]
            roles_list = {}
            for role in data["arguments"]:
                role_type = role["role"]
                role_start = role["argument_start_index"]
                role_text = role["argument"]
                role_end = role_start + len(role_text) - 1
                roles_list[role_type] = {
                    "role_type": role_type,
                    "start": role_start,
                    "end": role_end,
                    "argument": role_text
                }
            (sent_words, new_sent_words, ori_2_new_sent_index,
             new_roles_list) = process_sent_ori_2_new(sentence.lower(),
                                                      roles_list)
            new_sent_labels = [u"O"] * len(new_sent_words)
            for role_type, role in new_roles_list.items():
                for i in range(role["start"], role["end"] + 1):
                    if i == role["start"]:
                        new_sent_labels[i] = u"B-{}".format(role_type)
                    else:
                        new_sent_labels[i] = u"I-{}".format(role_type)
            example = Example(
                id=event_id,
                text_a=u" ".join(new_sent_words),
                label=u" ".join(new_sent_labels),
                ori_text=sent_words,
                ori_2_new_index=ori_2_new_sent_index,
                roles=new_roles_list,
                sentence=sentence)

            if k > 0:
                print(u"example {} : {}".format(
                    k, json.dumps(
                        example._asdict(), ensure_ascii=False)))
            k -= 1
            examples.append(example)
        return examples

    def _read_json_file(self, input_file):
        """_read_json_file"""
        input_data = []
        with open(input_file, "r", encoding='utf8') as f:
            for line in f:
                d_json = json.loads(line.strip())
                input_data.append(d_json)
        examples = self._process_examples_by_json(input_data)
        return examples

    def get_examples_by_file(self, input_file):
        """get_examples_by_file"""
        return self._read_json_file(input_file)

    def _pad_batch_records(self, batch_records):
        """_pad_batch_records"""
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [
            record.text_type_ids for record in batch_records
        ]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(
            batch_label_ids, pad_idx=len(self.label_map) - 1)
        padded_task_ids = np.ones_like(
            padded_token_ids, dtype="int64") * self.task_id

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            padded_task_ids, input_mask, padded_label_ids, batch_seq_lens
        ]
        return return_list

    def _reseg_token_label(self, tokens, labels, tokenizer):
        """_reseg_token_label"""
        assert len(tokens) == len(labels)
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            if len(sub_token) == 1:
                ret_labels.append(label)
                continue

            if label == "O" or label.startswith("I-"):
                ret_labels.extend([label] * len(sub_token))
            elif label.startswith("B-"):
                i_label = "I-" + label[2:]
                ret_labels.extend([label] + [i_label] * (len(sub_token) - 1))

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """_convert_example_to_record"""
        tokens = tokenization.whitespace_tokenize(example.text_a)
        labels = tokenization.whitespace_tokenize(example.label)
        tokens, labels = self._reseg_token_label(tokens, labels, tokenizer)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        position_ids = list(range(len(token_ids)))
        text_type_ids = [0] * len(token_ids)
        no_entity_id = len(self.label_map) - 1
        label_ids = [no_entity_id
                     ] + [self.label_map[label]
                          for label in labels] + [no_entity_id]

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_ids'])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_ids=label_ids)
        return record

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        k = 0
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.tokenizer)
            if k > 0:
                print(u"feature {} : {}".format(
                    k, json.dumps(
                        record._asdict(), ensure_ascii=False)))
            k -= 1
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def get_num_examples(self, input_file):
        """get_num_examples"""
        examples = self._read_json_file(input_file)
        return len(examples)

    def data_generator(self,
                       input_file,
                       batch_size,
                       epoch,
                       dev_count=1,
                       shuffle=True,
                       phase=None):
        """data_generator"""
        examples = self._read_json_file(input_file)

        def wrapper():
            """wrapper"""
            all_dev_batches = []
            for epoch_index in range(epoch):
                if phase == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, phase=phase):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        return wrapper
