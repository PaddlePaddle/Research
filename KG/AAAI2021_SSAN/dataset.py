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

from utils import tokenization


log = logging.getLogger(__name__)

if six.PY3:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def csv_reader(fd, delimiter='\t'):
    def gen():
        for i in fd:
            yield i.rstrip('\n').split(delimiter)
    return gen()


class BaseReader(object):
    def __init__(self,
                 vocab_path,
                 label_map_config=None,
                 max_seq_len=512,
                 max_ent_cnt=42,
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
        self.max_ent_cnt = max_ent_cnt
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
        self.ner_map = {'PAD': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}
        distance_buckets = np.zeros((512), dtype='int64')
        distance_buckets[1] = 1
        distance_buckets[2:] = 2
        distance_buckets[4:] = 3
        distance_buckets[8:] = 4
        distance_buckets[16:] = 5
        distance_buckets[32:] = 6
        distance_buckets[64:] = 7
        distance_buckets[128:] = 8
        distance_buckets[256:] = 9
        self.distance_buckets = distance_buckets

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

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

from dataclasses import dataclass
@dataclass(frozen=False)
class DocREDExample:

    guid: str
    title: str
    vertexSet: list
    sents: list
    labels: None


class DocREDReader(BaseReader):
    def _load_json(self, input_file):
        """Read DocRED json file into examples"""
        with open(input_file, 'r') as f:
            examples_raw = json.load(f)
        examples = []
        for (i, ins) in enumerate(examples_raw):
            guid = i
            examples.append(DocREDExample(guid=guid,
                                          title=ins['title'],
                                          vertexSet=ins['vertexSet'],
                                          sents=ins['sents'],
                                          labels=ins['labels'] if 'labels' in ins.keys() else None))
        return examples

    def get_num_train_examples(self, data_dir):
        examples = self._load_json(os.path.join(data_dir, "train_annotated.json"))
        return len(examples)

    def data_generator(self,
                       data_dir,
                       mode,
                       batch_size,
                       epoch,
                       dev_count=1):

        if mode == 'train':
            datafile = os.path.join(data_dir, "train_annotated.json")
            shuffle = True
        elif mode == 'eval':
            datafile = os.path.join(data_dir, "dev.json")
            shuffle = False
        elif mode == 'test':
            datafile = os.path.join(data_dir, "test.json")
            shuffle = False
        else:
            raise Exception("Invalid mode for data reader.")
        examples = self._load_json(datafile)

        def wrapper():
            all_dev_batches = []
            for epoch_index in range(epoch):
                if mode == "train":
                    self.current_example = 0
                    self.current_epoch = epoch_index
                if shuffle:
                    np.random.shuffle(examples)

                for batch_data in self._prepare_batch_data(
                        examples, batch_size, mode=mode):
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

    def _prepare_batch_data(self, examples, batch_size, mode=None):
        """generate batch records"""
        batch_records, max_len = [], 0
        for index, example in enumerate(examples):
            if mode == "train":
                self.current_example = index
            record = self._convert_example_to_record(example, self.max_seq_len,
                                                     self.max_ent_cnt, self.tokenizer)
            max_len = max(max_len, len(record.token_ids))
            if self.in_tokens:
                to_append = (len(batch_records) + 1) * max_len <= batch_size
            else:
                to_append = len(batch_records) < batch_size
            if to_append:
                batch_records.append(record)
            else:
                yield self._batch_records(batch_records)
                batch_records, max_len = [record], len(record.token_ids)

        # drop last batch!
        # if batch_records:
            # yield self._batch_records(batch_records)

    def _batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_input_mask = [record.input_mask for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_ent_mask = [record.ent_mask for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]
        batch_label_mask = [record.label_mask for record in batch_records]
        batch_ent_ner = [record.ent_ner for record in batch_records]
        batch_ent_pos = [record.ent_pos for record in batch_records]
        batch_ent_distance = [record.ent_distance for record in batch_records]
        batch_structure_mask = [record.structure_mask for record in batch_records]
        padded_task_ids = np.ones_like(batch_token_ids, dtype="int64") * self.task_id
        return_list = [
            batch_token_ids, batch_input_mask, batch_text_type_ids, batch_position_ids, padded_task_ids,
            batch_ent_mask, batch_label_ids, batch_label_mask, batch_ent_ner, batch_ent_pos,
            batch_ent_distance, batch_structure_mask
        ]
        return return_list

    def norm_mask(self, input_mask):
        output_mask = np.zeros(input_mask.shape)
        for i in range(len(input_mask)):
            if not np.all(input_mask[i] == 0):
                output_mask[i] = input_mask[i] / sum(input_mask[i])
        return output_mask

    def _convert_example_to_record(self, example, max_seq_length, max_ent_cnt, tokenizer):
        input_tokens = []
        tok_to_sent = []
        tok_to_word = []
        for sent_idx, sent in enumerate(example.sents):
            for word_idx, word in enumerate(sent):
                word = tokenization.convert_to_unicode(word)
                tokens_tmp = tokenizer.tokenize(word)
                input_tokens += tokens_tmp
                tok_to_sent += [sent_idx] * len(tokens_tmp)
                tok_to_word += [word_idx] * len(tokens_tmp)

        if len(input_tokens) <= max_seq_length - 2:
            input_tokens = ['[CLS]'] + input_tokens + ['[SEP]']
            tok_to_sent = [None] + tok_to_sent + [None]
            tok_to_word = [None] + tok_to_word + [None]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            input_mask = [1] * len(input_ids)
            text_type_ids = [0] * len(input_ids)
            position_ids = list(range(len(input_ids)))
            # padding
            padding = [None] * (max_seq_length - len(input_ids))
            tok_to_sent += padding
            tok_to_word += padding
            padding = [0] * (max_seq_length - len(input_ids))
            input_mask += padding
            text_type_ids += padding
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            position_ids += padding

        else:
            input_tokens = input_tokens[:max_seq_length - 2]
            tok_to_sent = tok_to_sent[:max_seq_length - 2]
            tok_to_word = tok_to_word[:max_seq_length - 2]
            input_tokens = ['[CLS]'] + input_tokens + ['[SEP]']
            tok_to_sent = [None] + tok_to_sent + [None]
            tok_to_word = [None] + tok_to_word + [None]
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            input_mask = [1] * len(input_ids)
            text_type_ids = [0] * len(input_ids)
            position_ids = list(range(len(input_ids)))

        # ent_mask & ner / coreference feature
        ent_mask = np.zeros((max_ent_cnt, max_seq_length), dtype='int64')
        ent_ner = [0] * max_seq_length
        ent_pos = [0] * max_seq_length
        tok_to_ent = [-1] * max_seq_length
        ents = example.vertexSet
        for ent_idx, ent in enumerate(ents):
            for mention in ent:
                for tok_idx in range(len(input_ids)):
                    if tok_to_sent[tok_idx] == mention['sent_id'] \
                            and mention['pos'][0] <= tok_to_word[tok_idx] < mention['pos'][1]:
                        ent_mask[ent_idx][tok_idx] = 1
                        ent_ner[tok_idx] = self.ner_map[ent[0]['type']]
                        ent_pos[tok_idx] = ent_idx + 1
                        tok_to_ent[tok_idx] = ent_idx

        # distance feature
        ent_first_appearance = [0] * max_ent_cnt
        ent_distance = np.zeros((max_ent_cnt, max_ent_cnt), dtype='int64')  # padding id is 10
        for i in range(len(ents)):
            if np.all(ent_mask[i] == 0):
                continue
            else:
                ent_first_appearance[i] = np.where(ent_mask[i] == 1)[0][0]
        for i in range(len(ents)):
            for j in range(len(ents)):
                if ent_first_appearance[i] != 0 and ent_first_appearance[j] != 0:
                    if ent_first_appearance[i] >= ent_first_appearance[j]:
                        ent_distance[i][j] = self.distance_buckets[ent_first_appearance[i] - ent_first_appearance[j]]
                    else:
                        ent_distance[i][j] = - self.distance_buckets[- ent_first_appearance[i] + ent_first_appearance[j]]
        ent_distance += 10  # norm from [-9, 9] to [1, 19]

        # structure prior for attentive biase
        # PRIOR DEFINITION  | share ent context |   diff ent context |    No ent
        # share sem context |    intra-coref    |    intra-relate    |    intra-NA
        # diff sem context  |    inter-coref    |    inter-relate    |
        structure_mask = np.zeros((5, max_seq_length, max_seq_length), dtype='float')
        for i in range(max_seq_length):
            if input_mask[i] == 0:
                break
            else:
                if tok_to_ent[i] != -1:
                    for j in range(max_seq_length):
                        if tok_to_sent[j] is None:
                            continue
                        #  intra
                        if tok_to_sent[j] == tok_to_sent[i]:
                            # intra-coref
                            if tok_to_ent[j] == tok_to_ent[i]:
                                structure_mask[0][i][j] = 1
                            # intra-relate
                            elif tok_to_ent[j] != -1:
                                structure_mask[1][i][j] = 1
                            # intra-NA
                            else:
                                structure_mask[2][i][j] = 1
                        else:
                            # inter-coref
                            if tok_to_ent[j] == tok_to_ent[i]:
                                structure_mask[3][i][j] = 1
                            # inter-relate
                            elif tok_to_ent[j] != -1:
                                structure_mask[4][i][j] = 1

        # label
        label_ids = np.zeros((max_ent_cnt, max_ent_cnt, len(self.label_map.keys())), dtype='int64')
        # test file does not have "labels"
        if example.labels is not None:
            labels = example.labels
            for label in labels:
                label_ids[label['h']][label['t']][self.label_map[label['r']]] = 1
        for h in range(len(ents)):
            for t in range(len(ents)):
                if np.all(label_ids[h][t] == 0):
                    label_ids[h][t][0] = 1

        label_mask = np.zeros((max_ent_cnt, max_ent_cnt), dtype='int64')
        label_mask[:len(ents), :len(ents)] = 1
        for ent in range(len(ents)):
            label_mask[ent][ent] = 0
        for ent in range(len(ents)):
            if np.all(ent_mask[ent] == 0):
                label_mask[ent, :] = 0
                label_mask[:, ent] = 0

        ent_mask = self.norm_mask(ent_mask)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(text_type_ids) == max_seq_length
        assert len(position_ids) == max_seq_length
        assert ent_mask.shape == (max_ent_cnt, max_seq_length)
        assert label_ids.shape == (max_ent_cnt, max_ent_cnt, len(self.label_map.keys()))
        assert label_mask.shape == (max_ent_cnt, max_ent_cnt)
        assert len(ent_ner) == max_seq_length
        assert len(ent_pos) == max_seq_length
        assert ent_distance.shape == (max_ent_cnt, max_ent_cnt)
        assert structure_mask.shape == (5, max_seq_length, max_seq_length)

        input_ids = np.expand_dims(input_ids, axis=-1).astype('int64')
        input_mask = np.expand_dims(input_mask, axis=-1).astype('int64')
        text_type_ids = np.expand_dims(text_type_ids, axis=-1).astype('int64')
        position_ids = np.expand_dims(position_ids, axis=-1).astype('int64')
        ent_ner = np.expand_dims(ent_ner, axis=-1).astype('int64')
        ent_pos = np.expand_dims(ent_pos, axis=-1).astype('int64')
        ent_distance = np.expand_dims(ent_distance, axis=-1).astype('int64')

        Record = namedtuple(
            'Record',
            ['token_ids', 'input_mask', 'text_type_ids', 'position_ids', 'ent_mask', 'label_ids',
             'label_mask', 'ent_ner', 'ent_pos', 'ent_distance', 'structure_mask'])
        record = Record(
            token_ids=input_ids,
            input_mask=input_mask,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            ent_mask=ent_mask,
            label_ids=label_ids,
            label_mask=label_mask,
            ent_ner=ent_ner,
            ent_pos=ent_pos,
            ent_distance=ent_distance,
            structure_mask=structure_mask)
        return record


if __name__ == '__main__':
    pass
