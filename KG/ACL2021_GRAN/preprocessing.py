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
Preprocess of the datasets.
"""

import collections
import json
import re
import os
import argparse


class JF17kPreprocess(object):
    """
    Preprocess of JF17k.
    """

    def __init__(self, raw_train, raw_test, train, test, vocab, ground_truth):
        self.raw_train = raw_train
        self.raw_test = raw_test
        self.train = train
        self.test = test
        self.vocab = vocab
        self.ground_truth = ground_truth

    def change_train_data_format(self):
        """
        Reformat training data.
        """
        max_arity = 0
        total_instance = 0
        with open(self.train, "w") as fw:
            with open(self.raw_train, "r") as fr:
                for line in fr.readlines():
                    tokens = line.strip().split("\t")
                    relation = tokens[0]
                    subject = tokens[1]
                    object = tokens[2]
                    arity = len(tokens) - 1

                    cnt = 2
                    new_obj = collections.OrderedDict()
                    new_obj['N'] = arity
                    new_obj['relation'] = relation
                    new_obj['subject'] = subject
                    new_obj['object'] = object
                    for idx in range(2, arity):
                        key = relation + "_" + str(idx - 2)
                        new_obj[key] = [tokens[idx + 1]]
                        cnt += 1
                    assert arity == cnt
                    if arity > max_arity:
                        max_arity = arity
                    total_instance += 1

                    fw.write(json.dumps(new_obj) + "\n")
        print "max train arity: %s" % max_arity
        print "number of train instances: %s" % total_instance

    def change_test_data_format(self):
        """
        Reformat test data.
        """
        max_arity = 0
        total_instance = 0
        with open(self.test, "w") as fw:
            with open(self.raw_test, "r") as fr:
                for line in fr.readlines():
                    tokens = line.strip().split("\t")
                    relation = tokens[1]
                    subject = tokens[2]
                    object = tokens[3]
                    arity = len(tokens) - 2

                    cnt = 2
                    new_obj = collections.OrderedDict()
                    new_obj['N'] = arity
                    new_obj['relation'] = relation
                    new_obj['subject'] = subject
                    new_obj['object'] = object
                    for idx in range(2, arity):
                        key = relation + "_" + str(idx - 2)
                        new_obj[key] = [tokens[idx + 2]]
                        cnt += 1
                    assert arity == cnt
                    if arity > max_arity:
                        max_arity = arity
                    total_instance += 1

                    fw.write(json.dumps(new_obj) + "\n")
        print "max test arity: %s" % max_arity
        print "number of test instances: %s" % total_instance

    def get_unique_roles_values(self):
        """
        Get unique roles (relations) and values (entities).
        """
        role_lst = dict()
        value_lst = dict()
        all_files = [self.train, self.test]
        for input_file in all_files:
            with open(input_file, "r") as fr:
                for line in fr.readlines():
                    obj = json.loads(line.strip())
                    # get unique roles and values
                    for role in obj.keys():
                        if role == "N":
                            continue
                        elif role == "relation":
                            role_lst[obj[role]] = len(role_lst)
                        elif role == "subject":
                            value_lst[obj[role]] = len(value_lst)
                        elif role == "object":
                            value_lst[obj[role]] = len(value_lst)
                        else:
                            role_lst[role] = len(role_lst)
                            for value in obj[role]:
                                value_lst[value] = len(value_lst)
        print "number of unique roles: %s" % len(role_lst)
        print "number of unique values: %s" % len(value_lst)

        return role_lst, value_lst

    def write_vocab(self, role_lst, value_lst):
        """
        Write vocabulary.
        The vocabulary starts with two special tokens [PAD] and [MASK],
        followed by (sorted) relations and (sorted) entities.
        """
        fout = open(self.vocab, "w")
        fout.write("[PAD]" + "\n")
        fout.write("[MASK]" + "\n")
        for r in sorted(role_lst.keys()):
            fout.write(r + "\n")
        for v in sorted(value_lst.keys()):
            fout.write(v + "\n")
        fout.close()

    def write_ground_truth(self):
        """
        Write all ground truth (required for filtered evaluation).
        """
        total_instance = 0
        n_ary_instance = 0
        all_files = [self.train, self.test]
        with open(self.ground_truth, "w") as fw:
            for input_file in all_files:
                with open(input_file, "r") as fr:
                    for line in fr.readlines():
                        fw.write(line.strip("\r \n") + "\n")
                        total_instance += 1
                        if json.loads(line)["N"] > 2:
                            n_ary_instance += 1
        print "number of all instances: %s" % total_instance
        print "number of n_ary instances: %s" % n_ary_instance

    def preprocess(self):
        self.change_train_data_format()
        self.change_test_data_format()

        role_lst, value_lst = self.get_unique_roles_values()
        self.write_vocab(role_lst=role_lst, value_lst=value_lst)
        self.write_ground_truth()


class WikipeoplePreprocess(object):
    """
    Preprocess of WikiPeople.
    """

    def __init__(self, raw_train, raw_dev, raw_test, train, dev, test, vocab,
                 ground_truth, train_dev):
        self.raw_train = raw_train
        self.raw_dev = raw_dev
        self.raw_test = raw_test
        self.train = train
        self.dev = dev
        self.test = test
        self.vocab = vocab
        self.ground_truth = ground_truth
        self.train_dev = train_dev

    def change_data_format(self, input_file, output_file, drop_literals=False):
        """
        Reformat the data and drop literals if necessary.
        """
        max_arity = 0
        total_instance = 0
        with open(output_file, "w") as fw:
            with open(input_file, "r") as fr:
                for line in fr.readlines():
                    relation_h = None
                    relation_t = None
                    subject = None
                    object = None
                    arity = 2

                    obj = json.loads(line.strip())
                    obj.pop("N")
                    for role in obj.keys():
                        if "_h" in role:
                            relation_h = role[0:-2]
                            subject = obj[role]
                        elif "_t" in role:
                            relation_t = role[0:-2]
                            object = obj[role]
                        else:
                            assert type(obj[role]) == list
                            for value in obj[role]:
                                if drop_literals and "http://" in value:
                                    continue
                                arity += 1
                    assert relation_h == relation_t, \
                        "Relations for the head and the tail should be identical."
                    if drop_literals and ("http://" in subject or
                                          "http://" in object):
                        continue

                    cnt = 2
                    new_obj = collections.OrderedDict()
                    new_obj['N'] = arity
                    new_obj['relation'] = relation_h
                    new_obj['subject'] = subject
                    new_obj['object'] = object

                    obj.pop(relation_h + "_h")
                    obj.pop(relation_t + "_t")
                    for role in obj.keys():
                        value_lst = []
                        for value in obj[role]:
                            if drop_literals and "http://" in value:
                                continue
                            value_lst.append(value)
                        if len(value_lst) > 0:
                            new_obj[role] = value_lst
                            cnt += len(value_lst)
                    assert arity == cnt
                    if arity > max_arity:
                        max_arity = arity
                    total_instance += 1

                    fw.write(json.dumps(new_obj) + "\n")
        print "max arity: %s" % max_arity
        print "total instance: %s" % total_instance

    def get_unique_roles_values(self):
        """
        Get unique roles (relations) and values (entities).
        """
        role_lst = dict()
        value_lst = dict()
        all_files = [self.train, self.dev, self.test]
        for input_file in all_files:
            with open(input_file, "r") as fr:
                for line in fr.readlines():
                    obj = json.loads(line.strip())
                    # get unique roles and values
                    for role in obj.keys():
                        if role == "N":
                            continue
                        elif role == "relation":
                            role_lst[obj[role]] = len(role_lst)
                        elif role == "subject":
                            value_lst[obj[role]] = len(value_lst)
                        elif role == "object":
                            value_lst[obj[role]] = len(value_lst)
                        else:
                            role_lst[role] = len(role_lst)
                            for value in obj[role]:
                                value_lst[value] = len(value_lst)
        print "number of unique roles: %s" % len(role_lst)
        print "number of unique values: %s" % len(value_lst)

        return role_lst, value_lst

    def write_vocab(self, role_lst, value_lst):
        """
        Write vocabulary.
        The vocabulary starts with two special tokens [PAD] and [MASK],
        followed by (sorted) relations and (sorted) entities.
        """
        fout = open(self.vocab, "w")
        fout.write("[PAD]" + "\n")
        fout.write("[MASK]" + "\n")
        for r in sorted(role_lst.keys()):
            fout.write(r + "\n")
        for v in sorted(value_lst.keys()):
            fout.write(v + "\n")
        fout.close()

    def write_ground_truth(self):
        """
        Write all ground truth (required for filtered evaluation).
        """
        total_instance = 0
        n_ary_instance = 0
        all_files = [self.train, self.dev, self.test]
        with open(self.ground_truth, "w") as fw:
            for input_file in all_files:
                with open(input_file, "r") as fr:
                    for line in fr.readlines():
                        fw.write(line.strip("\r \n") + "\n")
                        total_instance += 1
                        if json.loads(line)["N"] > 2:
                            n_ary_instance += 1
        print "number of all instances: %s" % total_instance
        print "number of n_ary instances: %s" % n_ary_instance

    def combine_train_dev(self):
        """
        Combine train and dev files for model training.
        """
        all_files = [self.train, self.dev]
        with open(self.train_dev, "w") as fw:
            for input_file in all_files:
                with open(input_file, "r") as fr:
                    for line in fr.readlines():
                        fw.write(line.strip("\r \n") + "\n")

    def preprocess(self, drop_literals=False):
        self.change_data_format(
            input_file=self.raw_train,
            output_file=self.train,
            drop_literals=drop_literals)
        self.change_data_format(
            input_file=self.raw_dev,
            output_file=self.dev,
            drop_literals=drop_literals)
        self.change_data_format(
            input_file=self.raw_test,
            output_file=self.test,
            drop_literals=drop_literals)

        role_lst, value_lst = self.get_unique_roles_values()
        self.write_vocab(role_lst=role_lst, value_lst=value_lst)
        self.write_ground_truth()
        self.combine_train_dev()


class JF17kNPreprocess(object):
    """
    Preprocess of JF17K-n, where n is the arity.
    """

    def __init__(self, arity, raw_train, raw_dev, raw_test, train, dev, test,
                 vocab, ground_truth, train_dev):
        self.arity = arity
        self.raw_train = raw_train
        self.raw_dev = raw_dev
        self.raw_test = raw_test
        self.train = train
        self.dev = dev
        self.test = test
        self.vocab = vocab
        self.ground_truth = ground_truth
        self.train_dev = train_dev

    def change_data_format(self, input_file, output_file):
        """
        Reformat the data.
        """
        max_arity = 0
        total_instance = 0
        with open(output_file, "w") as fw:
            with open(input_file, "r") as fr:
                for line in fr.readlines():
                    tokens = re.split(r'\t|\s', line.strip())
                    relation = tokens[0]
                    subject = tokens[1]
                    object = tokens[2]
                    arity = len(tokens) - 1

                    cnt = 2
                    new_obj = collections.OrderedDict()
                    new_obj['N'] = arity
                    new_obj['relation'] = relation
                    new_obj['subject'] = subject
                    new_obj['object'] = object
                    for idx in range(2, arity):
                        key = relation + "_" + str(idx - 2)
                        new_obj[key] = [tokens[idx + 1]]
                        cnt += 1
                    assert arity == cnt
                    if arity > max_arity:
                        max_arity = arity
                    total_instance += 1

                    fw.write(json.dumps(new_obj) + "\n")
        print "max arity: %s" % max_arity
        print "number of instances: %s" % total_instance

    def get_unique_roles_values(self):
        """
        Get unique roles (relations) and values (entities).
        """
        role_lst = dict()
        value_lst = dict()
        all_files = [self.train, self.dev, self.test]
        for input_file in all_files:
            with open(input_file, "r") as fr:
                for line in fr.readlines():
                    obj = json.loads(line.strip())
                    # get unique roles and values
                    for role in obj.keys():
                        if role == "N":
                            continue
                        elif role == "relation":
                            role_lst[obj[role]] = len(role_lst)
                        elif role == "subject":
                            value_lst[obj[role]] = len(value_lst)
                        elif role == "object":
                            value_lst[obj[role]] = len(value_lst)
                        else:
                            role_lst[role] = len(role_lst)
                            for value in obj[role]:
                                value_lst[value] = len(value_lst)
        print "number of unique roles: %s" % len(role_lst)
        print "number of unique values: %s" % len(value_lst)

        return role_lst, value_lst

    def write_vocab(self, role_lst, value_lst):
        """
        Write vocabulary.
        The vocabulary starts with two special tokens [PAD] and [MASK],
        followed by (sorted) relations and (sorted) entities.
        """
        fout = open(self.vocab, "w")
        fout.write("[PAD]" + "\n")
        fout.write("[MASK]" + "\n")
        for r in sorted(role_lst.keys()):
            fout.write(r + "\n")
        for v in sorted(value_lst.keys()):
            fout.write(v + "\n")
        fout.close()

    def write_ground_truth(self):
        """
        Write all ground truth (required for filtered evaluation).
        """
        total_instance = 0
        n_ary_instance = 0
        all_files = [self.train, self.dev, self.test]
        with open(self.ground_truth, "w") as fw:
            for input_file in all_files:
                with open(input_file, "r") as fr:
                    for line in fr.readlines():
                        fw.write(line.strip("\r \n") + "\n")
                        total_instance += 1
                        if json.loads(line)["N"] > 2:
                            n_ary_instance += 1
        print "number of all instances: %s" % total_instance
        print "number of n_ary instances: %s" % n_ary_instance

    def combine_train_dev(self):
        """
        Combine train and dev files for model training.
        """
        all_files = [self.train, self.dev]
        with open(self.train_dev, "w") as fw:
            for input_file in all_files:
                with open(input_file, "r") as fr:
                    for line in fr.readlines():
                        fw.write(line.strip("\r \n") + "\n")

    def preprocess(self):
        self.change_data_format(
            input_file=self.raw_train, output_file=self.train)
        self.change_data_format(input_file=self.raw_dev, output_file=self.dev)
        self.change_data_format(
            input_file=self.raw_test, output_file=self.test)

        role_lst, value_lst = self.get_unique_roles_values()
        self.write_vocab(role_lst=role_lst, value_lst=value_lst)
        self.write_ground_truth()
        self.combine_train_dev()


class WikipeopleNPreprocess(object):
    """
    Preprocess of WikiPeople-n, where n is the arity.
    """

    def __init__(self, all_ground_truth, arity, raw_train, raw_dev, raw_test,
                 train, dev, test, vocab, ground_truth, train_dev):
        self.all_ground_truth = all_ground_truth
        self.arity = arity
        self.raw_train = raw_train
        self.raw_dev = raw_dev
        self.raw_test = raw_test
        self.train = train
        self.dev = dev
        self.test = test
        self.vocab = vocab
        self.ground_truth = ground_truth
        self.train_dev = train_dev

    def get_nary_data(self):
        """
        Get instances with the specific arity.
        """
        instance_lst = {}
        with open(self.all_ground_truth, "r") as fr:
            for line in fr.readlines():
                obj = json.loads(line.strip())
                if obj["N"] == self.arity:
                    relation = obj["relation"]
                    subject = obj["subject"]
                    object = obj["object"]
                    value_lst = []
                    for role in obj.keys():
                        if role == "N" or role == "relation" \
                                or role == "subject" or role == "object":
                            continue
                        else:
                            for value in obj[role]:
                                value_lst.append(value)
                    value_lst = sorted(value_lst)
                    instance = relation + " "
                    instance += subject + " "
                    instance += object + " "
                    instance += " ".join(x for x in value_lst)
                    instance_lst[instance] = line.strip()
        print "{} instances with arity {}".format(
            len(instance_lst), self.arity)

        return instance_lst

    def change_data_format(self, input_file, output_file, instance_lst):
        """
        Reformat the data.
        """
        total_instance = 0
        with open(output_file, "w") as fw:
            with open(input_file, "r") as fr:
                for line in fr.readlines():
                    tokens = re.split(r'\t|\s', line.strip())
                    relation = tokens[0]
                    subject = tokens[1]
                    object = tokens[2]
                    value_lst = []
                    for idx in range(3, len(tokens)):
                        value_lst.append(tokens[idx])
                    value_lst = sorted(value_lst)
                    instance = relation + " "
                    instance += subject + " "
                    instance += object + " "
                    instance += " ".join(x for x in value_lst)
                    assert instance in instance_lst
                    fw.write(instance_lst[instance] + "\n")
                    total_instance += 1
        print "total instance: %s" % total_instance

    def get_unique_roles_values(self):
        """
        Get unique roles (relations) and values (entities).
        """
        role_lst = dict()
        value_lst = dict()
        all_files = [self.train, self.dev, self.test]
        for input_file in all_files:
            with open(input_file, "r") as fr:
                for line in fr.readlines():
                    obj = json.loads(line.strip())
                    # get unique roles and values
                    for role in obj.keys():
                        if role == "N":
                            continue
                        elif role == "relation":
                            role_lst[obj[role]] = len(role_lst)
                        elif role == "subject":
                            value_lst[obj[role]] = len(value_lst)
                        elif role == "object":
                            value_lst[obj[role]] = len(value_lst)
                        else:
                            role_lst[role] = len(role_lst)
                            for value in obj[role]:
                                value_lst[value] = len(value_lst)
        print "number of unique roles: %s" % len(role_lst)
        print "number of unique values: %s" % len(value_lst)

        return role_lst, value_lst

    def write_vocab(self, role_lst, value_lst):
        """
        Write vocabulary.
        The vocabulary starts with two special tokens [PAD] and [MASK],
        followed by (sorted) relations and (sorted) entities.
        """
        fout = open(self.vocab, "w")
        fout.write("[PAD]" + "\n")
        fout.write("[MASK]" + "\n")
        for r in sorted(role_lst.keys()):
            fout.write(r + "\n")
        for v in sorted(value_lst.keys()):
            fout.write(v + "\n")
        fout.close()

    def write_ground_truth(self):
        """
        Write all ground truth (required for filtered evaluation).
        """
        total_instance = 0
        n_ary_instance = 0
        all_files = [self.train, self.dev, self.test]
        with open(self.ground_truth, "w") as fw:
            for input_file in all_files:
                with open(input_file, "r") as fr:
                    for line in fr.readlines():
                        fw.write(line.strip("\r \n") + "\n")
                        total_instance += 1
                        if json.loads(line)["N"] > 2:
                            n_ary_instance += 1
        print "number of all instances: %s" % total_instance
        print "number of n_ary instances: %s" % n_ary_instance

    def combine_train_dev(self):
        """
        Combine train and dev files for model training.
        """
        all_files = [self.train, self.dev]
        with open(self.train_dev, "w") as fw:
            for input_file in all_files:
                with open(input_file, "r") as fr:
                    for line in fr.readlines():
                        fw.write(line.strip("\r \n") + "\n")

    def preprocess(self):
        instance_lst = self.get_nary_data()
        self.change_data_format(
            input_file=self.raw_train,
            output_file=self.train,
            instance_lst=instance_lst)
        self.change_data_format(
            input_file=self.raw_dev,
            output_file=self.dev,
            instance_lst=instance_lst)
        self.change_data_format(
            input_file=self.raw_test,
            output_file=self.test,
            instance_lst=instance_lst)

        role_lst, value_lst = self.get_unique_roles_values()
        self.write_vocab(role_lst=role_lst, value_lst=value_lst)
        self.write_ground_truth()
        self.combine_train_dev()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        default=None,
        choices=[
            'jf17k', 'wikipeople', 'wikipeople-', 'jf17k-3', 'jf17k-4',
            'wikipeople-3', 'wikipeople-4'
        ])
    args = parser.parse_args()

    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    task_dir = os.path.join(data_dir, args.task)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    if args.task == 'jf17k':
        raw_train = os.path.join(os.getcwd(), 'raw', args.task, 'train.txt')
        raw_test = os.path.join(os.getcwd(), 'raw', args.task, 'test.txt')

        train = os.path.join(task_dir, 'train.json')
        test = os.path.join(task_dir, 'test.json')

        vocab = os.path.join(task_dir, 'vocab.txt')
        ground_truth = os.path.join(task_dir, 'all.json')

        jf17k = JF17kPreprocess(
            raw_train=raw_train,
            raw_test=raw_test,
            train=train,
            test=test,
            vocab=vocab,
            ground_truth=ground_truth)
        jf17k.preprocess()

    if args.task == 'wikipeople':
        raw_train = os.path.join(os.getcwd(), 'raw', args.task,
                                 'n-ary_train.json')
        raw_dev = os.path.join(os.getcwd(), 'raw', args.task,
                               'n-ary_valid.json')
        raw_test = os.path.join(os.getcwd(), 'raw', args.task,
                                'n-ary_test.json')

        train = os.path.join(task_dir, 'train.json')
        dev = os.path.join(task_dir, 'valid.json')
        test = os.path.join(task_dir, 'test.json')

        vocab = os.path.join(task_dir, 'vocab.txt')
        ground_truth = os.path.join(task_dir, 'all.json')
        train_dev = os.path.join(task_dir, 'train+valid.json')

        wikipeople = WikipeoplePreprocess(
            raw_train=raw_train,
            raw_dev=raw_dev,
            raw_test=raw_test,
            train=train,
            dev=dev,
            test=test,
            vocab=vocab,
            ground_truth=ground_truth,
            train_dev=train_dev)
        wikipeople.preprocess(drop_literals=False)

    if args.task == 'wikipeople-':
        raw_train = os.path.join(os.getcwd(), 'raw',
                                 args.task.rstrip('-'), 'n-ary_train.json')
        raw_dev = os.path.join(os.getcwd(), 'raw',
                               args.task.rstrip('-'), 'n-ary_valid.json')
        raw_test = os.path.join(os.getcwd(), 'raw',
                                args.task.rstrip('-'), 'n-ary_test.json')

        train = os.path.join(task_dir, 'train.json')
        dev = os.path.join(task_dir, 'valid.json')
        test = os.path.join(task_dir, 'test.json')

        vocab = os.path.join(task_dir, 'vocab.txt')
        ground_truth = os.path.join(task_dir, 'all.json')
        train_dev = os.path.join(task_dir, 'train+valid.json')

        wikipeople = WikipeoplePreprocess(
            raw_train=raw_train,
            raw_dev=raw_dev,
            raw_test=raw_test,
            train=train,
            dev=dev,
            test=test,
            vocab=vocab,
            ground_truth=ground_truth,
            train_dev=train_dev)
        wikipeople.preprocess(drop_literals=True)

    if args.task == 'jf17k-3' or args.task == 'jf17k-4':
        arity = 3 if args.task == 'jf17k-3' else 4
        raw_train = os.path.join(os.getcwd(), 'raw', args.task, 'train.txt')
        raw_dev = os.path.join(os.getcwd(), 'raw', args.task, 'valid.txt')
        raw_test = os.path.join(os.getcwd(), 'raw', args.task, 'test.txt')

        train = os.path.join(task_dir, 'train.json')
        dev = os.path.join(task_dir, 'valid.json')
        test = os.path.join(task_dir, 'test.json')

        vocab = os.path.join(task_dir, 'vocab.txt')
        ground_truth = os.path.join(task_dir, 'all.json')
        train_dev = os.path.join(task_dir, 'train+valid.json')

        jf17k_n = JF17kNPreprocess(
            arity=arity,
            raw_train=raw_train,
            raw_dev=raw_dev,
            raw_test=raw_test,
            train=train,
            dev=dev,
            test=test,
            vocab=vocab,
            ground_truth=ground_truth,
            train_dev=train_dev)
        jf17k_n.preprocess()

    if args.task == 'wikipeople-3' or args.task == 'wikipeople-4':
        arity = 3 if args.task == 'wikipeople-3' else 4
        raw_train = os.path.join(os.getcwd(), 'raw', args.task, 'train.txt')
        raw_dev = os.path.join(os.getcwd(), 'raw', args.task, 'valid.txt')
        raw_test = os.path.join(os.getcwd(), 'raw', args.task, 'test.txt')

        train = os.path.join(task_dir, 'train.json')
        dev = os.path.join(task_dir, 'valid.json')
        test = os.path.join(task_dir, 'test.json')

        vocab = os.path.join(task_dir, 'vocab.txt')
        ground_truth = os.path.join(task_dir, 'all.json')
        train_dev = os.path.join(task_dir, 'train+valid.json')

        all_ground_truth = os.path.join(os.getcwd(),
                                        'data/wikipeople/all.json')

        wikipeople_n = WikipeopleNPreprocess(
            all_ground_truth=all_ground_truth,
            arity=arity,
            raw_train=raw_train,
            raw_dev=raw_dev,
            raw_test=raw_test,
            train=train,
            dev=dev,
            test=test,
            vocab=vocab,
            ground_truth=ground_truth,
            train_dev=train_dev)
        wikipeople_n.preprocess()
