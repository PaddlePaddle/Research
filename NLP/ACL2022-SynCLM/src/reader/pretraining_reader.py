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

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import gzip
import six

import paddle.fluid as fluid
import functools
from reader.batching import prepare_batch_data


class SynCLMDataReader(object):

    def __init__(self,
                 filelist,
                 tokenizer,
                 batch_size=4096,
                 in_tokens=True,
                 max_seq_len=512,
                 shuffle_files=True,
                 random_seed=1,
                 epoch=100,
                 voc_size=0,
                 is_test=False,
                 tree_max_sub_num=10,
                 tree_max_neg_num=10,
                 phrase_max_neg_num=10):

        self.filelist = filelist
        self.batch_size = batch_size
        self.in_tokens = in_tokens
        self.random_seed = random_seed
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.voc_size = voc_size
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id

        self.tree_max_sub_num = tree_max_sub_num
        self.tree_max_neg_num = tree_max_neg_num
        self.phrase_max_neg_num = phrase_max_neg_num

        self.input_slots = 5
        self.is_test = is_test

        self.trainer_id = 0
        self.trainer_nums = 1
        self.files = open(filelist).readlines()
        self.total_file = len(self.files)

        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False

        self.global_rng = np.random.RandomState(random_seed)
        if self.shuffle_files:
            if os.getenv("PADDLE_TRAINER_ID"):
                self.trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
            if os.getenv("PADDLE_TRAINERS_NUM"):
                self.trainer_nums = int(os.getenv("PADDLE_TRAINERS_NUM"))
            # renew total_file
            self.total_file = len(self.files) // self.trainer_nums * self.trainer_nums

            tmp_files = []
            for each in range(epoch):
                each_files = [i for i in self.files]
                self.global_rng.shuffle(each_files)
                tmp_files += each_files
            self.files = tmp_files
            # renew epochs
            self.epoch = len(self.files) // self.total_file * self.total_file

        assert self.total_file > 0, \
            "[Error] data_dir is empty or less than %d" % self.trainer_nums

        if self.in_tokens:
            assert self.batch_size > 100, "Current batch size means total token's number, \
                                       it should not be set to too small number."

    def get_progress(self):
        """return current progress of traning data
        """
        return self.current_epoch, self.current_file_index, self.total_file, self.current_file

    def parse_line(self, line, max_seq_len=512):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """
        line = line.strip()
        if '\t' in line:
            line = line.split('\t')[1]
        line = line.strip('\r\n').split(";")
        assert len(line) == self.input_slots, \
            "One sample must have %d fields!" % self.input_slots

        (token_ids, sent_ids, pos_ids, seg_labels, dp_ids) = line
        token_ids = [int(token) for token in token_ids.split(" ")]
        sent_ids = [int(token) for token in sent_ids.split(" ")]
        pos_ids = [int(token) for token in pos_ids.split(" ")]
        seg_labels = [int(seg_label) for seg_label in seg_labels.split(" ")]
        dp_ids = [int(dp_id) for dp_id in dp_ids.split(" ")]

        assert len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels) == len(dp_ids) , \
            "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels) == len(dp_ids)"

        if len(token_ids) > max_seq_len:
            return [
                token_ids[:max_seq_len - 1] + [self.sep_id], sent_ids[:max_seq_len], pos_ids[:max_seq_len],
                seg_labels[:max_seq_len - 1] + [-1], dp_ids[:max_seq_len - 1] + [-1]
            ]
        return [token_ids, sent_ids, pos_ids, seg_labels, dp_ids]

    def read_file(self, file):
        if file.endswith('.gz'):
            with gzip.open(file, "rt") as f:
                for line in f:
                    yield line
        else:
            with open(file, "r") as f:
                for line in f:
                    yield line

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def shuffle_samples(self, sample_generator, buffer=1000):
        samples = []
        try:
            while True:
                while len(samples) < buffer:
                    sample = next(sample_generator)
                    samples.append(sample)
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
                samples = []
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(samples) == 0:
                yield None
            else:
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample

    def data_generator(self):
        """
        data_generator
        """

        def wrapper():

            def reader():
                for epoch in range(self.epoch):
                    self.current_epoch = epoch + 1
                    files = self.files

                    # during training, data are sliced by trainers
                    if self.shuffle_files:
                        start = epoch * self.total_file
                        end = start + self.total_file
                        files = [
                            file_ for index, file_ in enumerate(self.files[start:end])
                            if index % self.trainer_nums == self.trainer_id
                        ]

                    for index, file_ in enumerate(files):
                        file_ = file_.strip()
                        self.current_file_index = (index + 1) * self.trainer_nums
                        self.current_file = file_

                        # sample_generator = self.read_file(file_)
                        sample_generator = fluid.io.xmap_readers(
                            functools.partial(self.parse_line, max_seq_len=self.max_seq_len),
                            functools.partial(self.read_file, file=file_), 8, 2000)()

                        if not self.is_test:
                            # shuffle buffered sample
                            sample_generator = self.shuffle_samples(sample_generator)

                        for sample in sample_generator:
                            if sample is None:
                                continue
                            yield sample

            def batch_reader(reader, batch_size):
                batch, total_token_num, max_len = [], 0, 0
                for parsed_line in reader():
                    token_ids, sent_ids, pos_ids, seg_labels, dp_ids = parsed_line
                    max_len = max(max_len, len(token_ids))
                    if self.in_tokens:
                        to_append = (len(batch) + 1) * max_len <= batch_size
                    else:
                        to_append = len(batch) < batch_size

                    if to_append:
                        batch.append(parsed_line)
                        total_token_num += len(token_ids)
                    else:
                        yield batch, total_token_num
                        batch, total_token_num, max_len = [parsed_line], len(token_ids), len(token_ids)

                if len(batch) > 0:
                    yield batch, total_token_num

            for batch_data, total_token_num in batch_reader(reader, self.batch_size):
                yield prepare_batch_data(batch_data,
                                         total_token_num,
                                         tree_max_sub_num=self.tree_max_sub_num,
                                         tree_max_neg_num=self.tree_max_neg_num,
                                         phrase_max_neg_num=self.phrase_max_neg_num,
                                         voc_size=self.voc_size,
                                         pad_id=self.pad_id,
                                         cls_id=self.cls_id,
                                         sep_id=self.sep_id,
                                         mask_id=self.mask_id,
                                         return_input_mask=True,
                                         return_max_len=False,
                                         return_num_token=False)

        return wrapper


if __name__ == "__main__":
    pass
