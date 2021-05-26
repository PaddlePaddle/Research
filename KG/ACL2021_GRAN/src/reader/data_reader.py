"""
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
"""
# yapf: disable
from __future__ import print_function
from __future__ import division

import json
import numpy as np
import collections
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())


class NaryExample(object):
    """
    A single training/test example of n-ary fact.
    """
    def __init__(self,
                 arity,
                 relation,
                 head,
                 tail,
                 auxiliary_info=None):
        """
        Construct NaryExample.

        Args:
            arity (mandatory): arity of a given fact
            relation (mandatory): primary relation
            head (mandatory): primary head entity (subject)
            tail (mandatory): primary tail entity (object)
            auxiliary_info (optional): auxiliary attribute-value pairs,
                with attributes and values sorted in alphabetical order
        """
        self.arity = arity
        self.relation = relation
        self.head = head
        self.tail = tail
        self.auxiliary_info = auxiliary_info


class NaryFeature(object):
    """
    A single set of features used for training/test.
    """
    def __init__(self,
                 feature_id,
                 example_id,
                 input_tokens,
                 input_ids,
                 input_mask,
                 mask_position,
                 mask_label,
                 mask_type,
                 arity):
        """
        Construct NaryFeature.

        Args:
            feature_id: unique feature id
            example_id: corresponding example id
            input_tokens: input sequence of tokens
            input_ids: input sequence of ids
            input_mask: input sequence mask
            mask_position: position of masked token
            mask_label: label of masked token
            mask_type: type of masked token,
                1 for entities (values) and -1 for relations (attributes)
            arity: arity of the corresponding example
        """
        self.feature_id = feature_id
        self.example_id = example_id
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.mask_position = mask_position
        self.mask_label = mask_label
        self.mask_type = mask_type
        self.arity = arity


def read_examples(input_file):
    """
    Read a n-ary json file into a list of NaryExample.
    """
    examples, total_instance = [], 0
    with open(input_file, "r") as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            assert "N" in obj.keys() \
                   and "relation" in obj.keys() \
                   and "subject" in obj.keys() \
                   and "object" in obj.keys(), \
                "There are 4 mandatory fields: N, relation, subject, and object."
            arity = obj["N"]
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()
                # store attributes in alphabetical order
                for attribute in sorted(obj.keys()):
                    if attribute == "N" \
                            or attribute == "relation" \
                            or attribute == "subject" \
                            or attribute == "object":
                        continue
                    # store corresponding values in alphabetical order
                    auxiliary_info[attribute] = sorted(obj[attribute])
            """
            if len(examples) % 1000 == 0:
                logger.debug("*** Example ***")
                logger.debug("arity: %s" % str(arity))
                logger.debug("relation: %s" % relation)
                logger.debug("head: %s" % head)
                logger.debug("tail: %s" % tail)
                if auxiliary_info:
                    for attribute in auxiliary_info.keys():
                        logger.debug("attribute: %s" % attribute)
                        logger.debug("value(s): %s" % " ".join(
                            [value for value in auxiliary_info[attribute]]))
            """

            example = NaryExample(
                arity=arity,
                relation=relation,
                head=head,
                tail=tail,
                auxiliary_info=auxiliary_info)
            examples.append(example)
            total_instance += (2 * (arity - 2) + 3)

    return examples, total_instance


def convert_examples_to_features(examples, vocabulary, max_arity, max_seq_length):
    """
    Convert a set of NaryExample into a set of NaryFeature. Each single
    NaryExample is converted into (2*(n-2)+3) NaryFeature, where n is
    the arity of the given example.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, " \
        "and max_aux attribute-value pairs."

    features = []
    feature_id = 0
    for (example_id, example) in enumerate(examples):
        # get original input tokens and input mask
        rht = [example.relation, example.head, example.tail]
        rht_mask = [1, 1, 1]

        aux_attributes = []
        aux_attributes_mask = []
        aux_values = []
        aux_values_mask = []
        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_attributes.append(attribute)
                    aux_values.append(value)
                    aux_attributes_mask.append(1)
                    aux_values_mask.append(1)

        while len(aux_attributes) < max_aux:
            aux_attributes.append("[PAD]")
            aux_values.append("[PAD]")
            aux_attributes_mask.append(0)
            aux_values_mask.append(0)
        assert len(aux_attributes) == max_aux
        assert len(aux_values) == max_aux
        assert len(aux_attributes_mask) == max_aux
        assert len(aux_values_mask) == max_aux

        orig_input_tokens = rht + aux_attributes + aux_values
        orig_input_mask = rht_mask + aux_attributes_mask + aux_values_mask
        assert len(orig_input_tokens) == max_seq_length
        assert len(orig_input_mask) == max_seq_length

        # generate a feature by masking each of the tokens
        for mask_position in range(max_seq_length):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue
            mask_label = vocabulary.vocab[orig_input_tokens[mask_position]]
            mask_type = -1 if mask_position == 0 or \
                              2 < mask_position < max_aux + 3 else 1

            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"
            input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
            assert len(input_tokens) == max_seq_length
            assert len(input_ids) == max_seq_length
            """
            if example_id == 275889:
                logger.debug("*** Feature ***")
                logger.debug("feature id: %s" % feature_id)
                logger.debug("example id: %s" % example_id)
                logger.debug("input tokens: %s" % " ".join(
                    [x for x in input_tokens]))
                logger.debug("input ids: %s" % " ".join(
                    [str(x) for x in input_ids]))
                logger.debug("input mask: %s" % " ".join(
                    [str(x) for x in orig_input_mask]))
                logger.debug("mask position: %s" % mask_position)
                logger.debug("mask label: %s" % mask_label)
                logger.debug("mask type: %s" % mask_type)
            """

            feature = NaryFeature(
                feature_id=feature_id,
                example_id=example_id,
                input_tokens=input_tokens,
                input_ids=input_ids,
                input_mask=orig_input_mask,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                arity=example.arity)
            features.append(feature)
            feature_id += 1

    return features


class DataReader(object):
    """
    DataReader class.
    """

    def __init__(self,
                 data_path,
                 max_arity=2,
                 max_seq_length=3,
                 batch_size=4096,
                 is_training=True,
                 shuffle=True,
                 dev_count=1,
                 epoch=10):
        self.max_arity = max_arity
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

        self.is_training = is_training
        self.shuffle = shuffle
        self.dev_count = dev_count
        self.epoch = epoch
        if not is_training:
            self.shuffle = False
            self.dev_count = 1
            self.epoch = 1

        self.examples, self.total_instance = read_examples(data_path)
        self.current_epoch = -1
        self.current_instance_index = -1

    def get_progress(self):
        """
        Get progress for training phase.
        """
        return self.current_instance_index, self.current_epoch

    def get_features(self, vocabulary):
        """
        Get training/test features.
        """
        features = convert_examples_to_features(
            examples=self.examples,
            vocabulary=vocabulary,
            max_arity=self.max_arity,
            max_seq_length=self.max_seq_length)
        return features

    def data_generator(self, vocabulary):
        """
        Data generator used during training/test.
        """

        def wrapper():
            """
            Wrapper batch data.
            """

            def batch_reader():
                """
                Read features into batches, where each instance is stored as:
                [input_ids, input_mask, mask_position, mask_label, mask_type].
                """
                batch = []
                for epoch_index in range(self.epoch):
                    self.current_epoch = epoch_index
                    if self.shuffle is True:
                        np.random.shuffle(self.examples)
                    features = self.get_features(vocabulary=vocabulary)

                    for (index, feature) in enumerate(features):
                        self.current_instance_index = index
                        input_ids = feature.input_ids
                        input_mask = feature.input_mask
                        mask_position = feature.mask_position
                        mask_label = feature.mask_label
                        mask_type = feature.mask_type
                        feature_out = [input_ids] + [input_mask] + \
                                      [mask_position] + [mask_label] + [mask_type]
                        to_append = len(batch) < self.batch_size
                        if to_append is False:
                            yield batch
                            batch = [feature_out]
                        else:
                            batch.append(feature_out)
                if len(batch) > 0:
                    yield batch

            all_dev_batches = []
            for batch_data in batch_reader():
                batch_data = prepare_batch_data(
                    batch_data,
                    max_arity=self.max_arity,
                    max_seq_length=self.max_seq_length)
                if len(all_dev_batches) < self.dev_count:
                    all_dev_batches.append(batch_data)

                if len(all_dev_batches) == self.dev_count:
                    for batch in all_dev_batches:
                        yield batch
                    all_dev_batches = []

        return wrapper


def prepare_batch_data(insts, max_arity, max_seq_length):
    """
    Format batch input for training/test. Output a list of six entries:
        return_list[0]: batch_input_ids (batch_size * max_seq_length * 1)
        return_list[1]: batch_input_mask (batch_size * max_seq_length * 1)
        return_list[2]: batch_mask_position (batch_size * 1)
        return_list[3]: batch_mask_label (batch_size * 1)
        return_list[4]: batch_mask_type (batch_size * 1)
        return_list[5]: edge_labels (max_seq_length * max_seq_length * 1)
    Note: mask_position indicates positions in a batch (not individual instances).
    And edge_labels are shared across all instances in the batch.
    """
    batch_input_ids = np.array([
        inst[0] for inst in insts
    ]).astype("int64").reshape([-1, max_seq_length, 1])
    batch_input_mask = np.array([
        inst[1] for inst in insts
    ]).astype("float32").reshape([-1, max_seq_length, 1])

    batch_mask_position = np.array([
        idx * max_seq_length + inst[2] for (idx, inst) in enumerate(insts)
    ]).astype("int64").reshape([-1, 1])
    batch_mask_label = np.array(
        [inst[3] for inst in insts]).astype("int64").reshape([-1, 1])
    batch_mask_type = np.array(
        [inst[4] for inst in insts]).astype("int64").reshape([-1, 1])

    # edge labels between input nodes (used for GRAN-hete)
    #     0: no edge
    #     1: relation-subject
    #     2: relation-object
    #     3: relation-attribute
    #     4: attribute-value
    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 2] + [3] * max_aux + [0] * max_aux)
    edge_labels.append([1] + [0] * (max_seq_length - 1))
    edge_labels.append([2] + [0] * (max_seq_length - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [3, 0, 0] + [0] * max_aux + [0] * idx + [4] + [0] * (max_aux - idx - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [0, 0, 0] + [0] * idx + [4] + [0] * (max_aux - idx - 1) + [0] * max_aux)
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])
    """
    # unlabeled edges between input nodes (used for GRAN-homo)
    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 1] + [1] * max_aux + [0] * max_aux)
    edge_labels.append([1] + [0] * (max_seq_length - 1))
    edge_labels.append([1] + [0] * (max_seq_length - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [1, 0, 0] + [0] * max_aux + [0] * idx + [1] + [0] * (max_aux - idx - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [0, 0, 0] + [0] * idx + [1] + [0] * (max_aux - idx - 1) + [0] * max_aux)
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])
    """
    """
    # no edges between input nodes (used for GRAN-complete)
    edge_labels = [[0] * max_seq_length] * max_seq_length
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])
    """

    return_list = [batch_input_ids] + [batch_input_mask] + \
                  [batch_mask_position] + [batch_mask_label] + [batch_mask_type] + \
                  [edge_labels]
    return return_list
