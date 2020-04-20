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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os
import random
import sys
from collections import namedtuple
from io import open

import numpy as np
import six
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
    """BaseReader"""

    def __init__(
        self,
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
        task_id=0,
    ):
        """init"""
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case,
        )
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

    def get_num_examples(self, input_file):
        """get_num_examples"""
        examples = self._read_tsv(input_file)
        return len(examples)

    def data_generator(
        self,
        input_file,
        batch_size,
        epoch,
        dev_count=1,
        shuffle=True,
        phase=None,
    ):
        """data_generator"""
        examples = self._read_tsv(input_file)

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
                        examples, batch_size, phase=phase,
                ):
                    if len(all_dev_batches) < dev_count:
                        all_dev_batches.append(batch_data)
                    if len(all_dev_batches) == dev_count:
                        for batch in all_dev_batches:
                            yield batch
                        all_dev_batches = []

        def f():
            try:
                for i in wrapper():
                    yield i
            except Exception as e:
                import traceback
                traceback.print_exc()
        return f


class RankReader(BaseReader):
    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)

            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        if self.for_cn:
                            line[index] = text.replace(' ', '')
                        else:
                            line[index] = text
                # print(line)
                example = Example(*line)
                examples.append(example)
            return examples

    def _prepare_batch_data(self, examples, batch_size, phase=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        logging.info('prepare_batch_data')
        for index, example in enumerate(examples):
            if phase == "train":
                self.current_example = index
            record = self._convert_example_to_record(
                example, self.max_seq_len,
                self.tokenizer,
            )
            max_len = max(max_len, len(record.token_query_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._pad_batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_query_ids)

        if batch_records:
            yield self._pad_batch_records(batch_records)

    def _pad_batch_records(self, batch_records):
        """_pad_batch_records"""
        batch_token_query_ids = [
            record.token_query_ids for record in batch_records
        ]
        batch_text_type_query_ids = [
            record.text_type_query_ids for record in batch_records
        ]
        batch_position_query_ids = [
            record.position_query_ids for record in batch_records
        ]
        batch_token_left_ids = [
            record.token_left_ids for record in batch_records
        ]
        batch_text_type_left_ids = [
            record.text_type_left_ids for record in batch_records
        ]
        batch_position_left_ids = [
            record.position_left_ids for record in batch_records
        ]
        batch_token_right_ids = [
            record.token_right_ids for record in batch_records
        ]
        batch_text_type_right_ids = [
            record.text_type_right_ids for record in batch_records
        ]
        batch_position_right_ids = [
            record.position_right_ids for record in batch_records
        ]

        if batch_records[0].qid:
            batch_qids = [record.qid for record in batch_records]
            batch_qids = np.array(batch_qids).astype("int64").reshape(
                [-1, 1],
            )
        else:
            batch_qids = np.array([]).astype("int64").reshape([-1, 1])

        if not self.is_inference:
            batch_labels = [record.label_id for record in batch_records]
            batch_types = [record.type_id for record in batch_records]
            if self.is_classify:
                batch_labels = np.array(batch_labels).astype("int64").reshape(
                    [-1, 1],
                )
                batch_types = np.array(batch_types).astype("int64").reshape(
                    [-1, 1],
                )
            elif self.is_regression:
                batch_labels = np.array(batch_labels).astype("float32").reshape(
                    [-1, 1],
                )
        else:
            if batch_records[0].ent_id:
                batch_ent_ids = [record.ent_id for record in batch_records]
                batch_ent_ids = np.array(batch_ent_ids).reshape(
                    [-1, 1],
                )
            else:
                batch_ent_ids = np.array([]).reshape([-1, 1])

        # padding
        padded_token_query_ids, input_query_mask = pad_batch_data(
            batch_token_query_ids, pad_idx=self.pad_id, return_input_mask=True,
        )
        padded_text_type_query_ids = pad_batch_data(
            batch_text_type_query_ids, pad_idx=self.pad_id,
        )
        padded_position_query_ids = pad_batch_data(
            batch_position_query_ids, pad_idx=self.pad_id,
        )
        padded_task_query_ids = np.ones_like(
            padded_token_query_ids, dtype="int64",
        ) * self.task_id

        padded_token_left_ids, input_left_mask = pad_batch_data(
            batch_token_left_ids, pad_idx=self.pad_id, return_input_mask=True,
        )
        padded_text_type_left_ids = pad_batch_data(
            batch_text_type_left_ids, pad_idx=self.pad_id,
        )
        padded_position_left_ids = pad_batch_data(
            batch_position_left_ids, pad_idx=self.pad_id,
        )
        padded_task_left_ids = np.ones_like(
            padded_token_left_ids, dtype="int64",
        ) * self.task_id

        padded_token_right_ids, input_right_mask = pad_batch_data(
            batch_token_right_ids, pad_idx=self.pad_id, return_input_mask=True,
        )
        padded_text_type_right_ids = pad_batch_data(
            batch_text_type_right_ids, pad_idx=self.pad_id,
        )
        padded_position_right_ids = pad_batch_data(
            batch_position_right_ids, pad_idx=self.pad_id,
        )
        padded_task_right_ids = np.ones_like(
            padded_token_right_ids, dtype="int64",
        ) * self.task_id

        return_list = [
            padded_token_query_ids, padded_text_type_query_ids, padded_position_query_ids,
            padded_task_query_ids, input_query_mask,
            padded_token_left_ids, padded_text_type_left_ids, padded_position_left_ids,
            padded_task_left_ids, input_left_mask,
            padded_token_right_ids, padded_text_type_right_ids, padded_position_right_ids,
            padded_task_right_ids, input_right_mask,
        ]
        if not self.is_inference:
            return_list += [batch_labels, batch_types, batch_qids]
        else:
            return_list += [batch_qids, batch_ent_ids]
        return return_list

    def _truncate_seq_pair(self, tokens_a, tokens_b, tokens_c, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_a) + \
                len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
                tokens_a.pop()
            if len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
                tokens_b.pop()
            else:
                tokens_c.pop()

    def _convert_example_to_record(self, example, max_seq_length, tokenizer):
        """Converts a single `Example` into a single `Record`."""

        text_a = tokenization.convert_to_unicode(example.text_a)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None

        has_text_b = False
        if isinstance(example, dict):
            has_text_b = "text_b" in example.keys()
            has_text_c = "text_c" in example.keys()
        else:
            has_text_b = "text_b" in example._fields
            has_text_c = "text_c" in example._fields

        if has_text_b:
            text_b = tokenization.convert_to_unicode(example.text_b)
            tokens_b = tokenizer.tokenize(text_b)
        if has_text_c:
            text_c = tokenization.convert_to_unicode(example.text_c)
            tokens_c = tokenizer.tokenize(text_c)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
        if len(tokens_b) > max_seq_length - 2:
            tokens_b = tokens_b[0:(max_seq_length - 2)]
        if len(tokens_c) > max_seq_length - 2:
            tokens_c = tokens_c[0:(max_seq_length - 2)]

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
        tokens_query = []
        text_type_query_ids = []
        tokens_query.append("[CLS]")
        text_type_query_ids.append(0)
        for token in tokens_a:
            tokens_query.append(token)
            text_type_query_ids.append(0)
        tokens_query.append("[SEP]")
        text_type_query_ids.append(0)

        tokens_left = []
        text_type_left_ids = []
        tokens_left.append("[CLS]")
        text_type_left_ids.append(0)
        for token in tokens_b:
            tokens_left.append(token)
            text_type_left_ids.append(0)
        tokens_left.append("[SEP]")
        text_type_left_ids.append(0)

        tokens_right = []
        text_type_right_ids = []
        tokens_right.append("[CLS]")
        text_type_right_ids.append(0)
        for token in tokens_c:
            tokens_right.append(token)
            text_type_right_ids.append(0)
        tokens_right.append("[SEP]")
        text_type_right_ids.append(0)

        token_query_ids = tokenizer.convert_tokens_to_ids(tokens_query)
        position_query_ids = list(range(len(token_query_ids)))
        token_left_ids = tokenizer.convert_tokens_to_ids(tokens_left)
        position_left_ids = list(range(len(token_left_ids)))
        token_right_ids = tokenizer.convert_tokens_to_ids(tokens_right)
        position_right_ids = list(range(len(token_right_ids)))

        if self.is_inference:
            Record = namedtuple(
                'Record',
                [
                    'token_query_ids', 'text_type_query_ids', 'position_query_ids',
                    'token_left_ids', 'text_type_left_ids', 'position_left_ids',
                    'token_right_ids', 'text_type_right_ids', 'position_right_ids',
                    'qid', 'ent_id',
                ],
            )
            qid, ent_id = None, None
            if 'qid' in example._fields:
                qid = example.qid
            if 'ent_id_b' in example._fields:
                ent_id = example.ent_id_b
            record = Record(
                token_query_ids=token_query_ids,
                text_type_query_ids=text_type_query_ids,
                position_query_ids=position_query_ids,
                token_left_ids=token_left_ids,
                text_type_left_ids=text_type_left_ids,
                position_left_ids=position_left_ids,
                token_right_ids=token_right_ids,
                text_type_right_ids=text_type_right_ids,
                position_right_ids=position_right_ids,
                qid=qid,
                ent_id=ent_id,
            )
        else:
            if self.label_map:
                type_id = self.label_map[example.type]
            else:
                type_id = example.type

            label_id = example.label
            Record = namedtuple(
                'Record',
                [
                    'token_query_ids', 'text_type_query_ids', 'position_query_ids',
                    'token_left_ids', 'text_type_left_ids', 'position_left_ids',
                    'token_right_ids', 'text_type_right_ids', 'position_right_ids',
                    'label_id', 'type_id', 'qid',
                ],
            )

            qid = None
            if "qid" in example._fields:
                qid = example.qid

            record = Record(
                token_query_ids=token_query_ids,
                text_type_query_ids=text_type_query_ids,
                position_query_ids=position_query_ids,
                token_left_ids=token_left_ids,
                text_type_left_ids=text_type_left_ids,
                position_left_ids=position_left_ids,
                token_right_ids=token_right_ids,
                text_type_right_ids=text_type_right_ids,
                position_right_ids=position_right_ids,
                label_id=label_id,
                type_id=type_id,
                qid=qid,
            )

        return record


if __name__ == '__main__':
    pass
