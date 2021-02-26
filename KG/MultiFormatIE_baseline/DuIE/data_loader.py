#!/usr/bin/env python
# coding=utf-8
"""
@File:	data_loader.py
@Time:	2021/01/27 14:23:19
@Author:	Benfeng Xu(xubenfeng@baidu.com) & Huapeng Qin(qinhuapeng@baidu.com)
@Desc:	Data loader implementation for DuIE baseline task.
"""
import collections
import json
import os
from typing import Optional, List, Union, Dict
from tqdm import tqdm

import numpy as np
import paddle
from dataclasses import dataclass

from paddlenlp.transformers import ErnieTokenizer

from extract_chinese_and_punct import ChineseAndPunctuationExtractor

InputFeature = collections.namedtuple("InputFeature", ["input_ids", "tok_to_orig_start_index", "tok_to_orig_end_index", "labels"])


def convert_example_to_feature(
    example, tokenizer: ErnieTokenizer, chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
    label_map , max_length: Optional[int] = 512, pad_to_max_length: Optional[bool] = None
):

    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = []
    buff = ""
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer._tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    num_labels = 2 *(len(label_map.keys()) - 2) + 2 # 2 tags for each predicate + I tag + O tag
    labels = [[0] * num_labels for i in range(len(tokens))]  # initialize tag
    #  find all entities and tag them with corresponding "B"/"I" labels
    if spo_list is not None:
        for spo in spo_list:
            for spo_object in spo['object'].keys():
                # assign relation label
                if spo['predicate'] in label_map.keys():
                    # simple relation
                    label_subject = label_map[spo['predicate']]
                    label_object = label_subject + 55
                    subject_sub_tokens = tokenizer._tokenize(spo['subject'])
                    object_sub_tokens = tokenizer._tokenize(spo['object']['@value'])
                else:
                    # complex relation
                    label_subject = label_map[spo['predicate'] + '_' + spo_object]
                    label_object = label_subject + 55
                    subject_sub_tokens = tokenizer._tokenize(spo['subject'])
                    object_sub_tokens = tokenizer._tokenize(spo['object'][spo_object])

                # assign token label
                # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
                # to prevent single token from being labeled into two different entity
                # we tag the longer entity first, then match the shorter entity within the rest text
                forbidden_index = None
                if len(subject_sub_tokens) > len(object_sub_tokens):
                    for index in range(
                            len(tokens) - len(subject_sub_tokens) + 1):
                        if tokens[index:index + len(
                                subject_sub_tokens)] == subject_sub_tokens:
                            labels[index][label_subject] = 1
                            for i in range(len(subject_sub_tokens) - 1):
                                labels[index + i + 1][1] = 1
                            forbidden_index = index
                            break

                    for index in range(
                            len(tokens) - len(object_sub_tokens) + 1):
                        if tokens[index:index + len(
                                object_sub_tokens)] == object_sub_tokens:
                            if forbidden_index is None:
                                labels[index][label_object] = 1
                                for i in range(len(object_sub_tokens) - 1):
                                    labels[index + i + 1][1] = 1
                                break
                            # check if labeled already
                            elif index < forbidden_index or index >= forbidden_index + len(
                                    subject_sub_tokens):
                                labels[index][label_object] = 1
                                for i in range(len(object_sub_tokens) - 1):
                                    labels[index + i + 1][1] = 1
                                break

                else:
                    for index in range(
                            len(tokens) - len(object_sub_tokens) + 1):
                        if tokens[index:index + len(
                                object_sub_tokens)] == object_sub_tokens:
                            labels[index][label_object] = 1
                            for i in range(len(object_sub_tokens) - 1):
                                labels[index + i + 1][1] = 1
                            forbidden_index = index
                            break

                    for index in range(
                            len(tokens) - len(subject_sub_tokens) + 1):
                        if tokens[index:index + len(
                                subject_sub_tokens)] == subject_sub_tokens:
                            if forbidden_index is None:
                                labels[index][label_subject] = 1
                                for i in range(len(subject_sub_tokens) - 1):
                                    labels[index + i + 1][1] = 1
                                break
                            elif index < forbidden_index or index >= forbidden_index + len(
                                    object_sub_tokens):
                                labels[index][label_subject] = 1
                                for i in range(len(subject_sub_tokens) - 1):
                                    labels[index + i + 1][1] = 1
                                break

        # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
        for i in range(len(labels)):
            if labels[i] == [0] * num_labels:
                labels[i][0] = 1

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if len(tokens) > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        tok_to_orig_start_index = tok_to_orig_start_index[0:(max_length - 2)]
        tok_to_orig_end_index = tok_to_orig_end_index[0:(max_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    outside_label = [[1] + [0] * (num_labels - 1)]
    labels = outside_label + labels + outside_label
    tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
    tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
    if len(tokens) < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - len(tokens))
        labels = labels + outside_label * (max_length - len(labels))
        tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (max_length - len(tok_to_orig_start_index))
        tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (max_length - len(tok_to_orig_end_index))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    assert len(token_ids) == max_length
    assert len(tok_to_orig_start_index) == max_length
    assert len(tok_to_orig_end_index) == max_length
    assert len(labels) == max_length

    return InputFeature(
        input_ids=np.array(token_ids, dtype=np.int16),
        tok_to_orig_start_index=np.array(tok_to_orig_start_index, dtype=np.int16),
        tok_to_orig_end_index=np.array(tok_to_orig_end_index, dtype=np.int16),
        labels=np.array(labels, dtype=np.int16)
    )


class DuIEDataset(paddle.io.Dataset):
    """
    Dataset of DuIE.
    """

    def __init__(
        self,
        input_ids: List[Union[List[int], np.ndarray]],
        tok_to_orig_start_index: List[Union[List[int], np.ndarray]],
        tok_to_orig_end_index: List[Union[List[int], np.ndarray]],
        labels: List[Union[List[int], np.ndarray, List[str], List[Dict]]]
    ):
        super(DuIEDataset, self).__init__()

        self.input_ids = input_ids
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.labels = labels

    def __len__(self):
        if isinstance(self.input_ids, np.ndarray):
            return self.input_ids.shape[0]
        else:
            return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": np.array(self.input_ids[item], dtype=np.int64),
            "tok_to_orig_start_index": np.array(self.tok_to_orig_start_index[item], dtype=np.int64),
            "tok_to_orig_end_index": np.array(self.tok_to_orig_end_index[item], dtype=np.int64),
            # If model inputs is generated in `collate_fn`, delete the data type casting.
            "labels": np.array(self.labels[item], dtype=np.int64),
        }

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, os.PathLike],
        tokenizer: ErnieTokenizer,
        max_length: Optional[int] = 512,
        pad_to_max_length: Optional[bool] = None
    ):
        assert os.path.exists(file_path) and os.path.isfile(file_path), f"{file_path} dose not exists or is not a file."
        label_map_path = os.path.join(os.path.dirname(file_path), "predicate2id.json")
        assert os.path.exists(label_map_path) and os.path.isfile(label_map_path), f"{label_map_path} dose not exists or is not a file."
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp)
        chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

        input_ids, tok_to_orig_start_index, tok_to_orig_end_index, labels = ([] for _ in range(4))
        dataset_scale = sum(1 for line in open(file_path, 'r'))
        with open(file_path, "r", encoding="utf-8") as fp:
            for line in tqdm(fp, total=dataset_scale):
                example = json.loads(line)
                input_feature = convert_example_to_feature(example, tokenizer, chineseandpunctuationextractor, label_map, max_length, pad_to_max_length)
                input_ids.append(input_feature.input_ids)
                tok_to_orig_start_index.append(input_feature.tok_to_orig_start_index)
                tok_to_orig_end_index.append(input_feature.tok_to_orig_end_index)
                labels.append(input_feature.labels)
        return cls(input_ids, tok_to_orig_start_index, tok_to_orig_end_index, labels)


@dataclass
class DataCollator:
    """
    Collator for DuIE.
    """
    tokenizer: ErnieTokenizer

    def __call__(
        self, examples: List[Dict[str, Union[list, np.ndarray]]]
    ):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])
        tok_to_orig_start_index = np.stack([x['tok_to_orig_start_index'] for x in examples])
        tok_to_orig_end_index = np.stack([x['tok_to_orig_end_index'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])

        # if isinstance(examples[0], dict):
        #     batch = self.tokenizer.pad(examples, return_tensors="np")
        return (
            batched_input_ids, tok_to_orig_start_index, tok_to_orig_end_index, labels
        )


if __name__ == "__main__":
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    d = DuIEDataset.from_file("./data/train_data.json", tokenizer)
    sampler = paddle.io.RandomSampler(data_source=d)
    batch_sampler = paddle.io.BatchSampler(sampler=sampler, batch_size=2)

    collator = DataCollator(tokenizer=tokenizer)
    loader = paddle.io.DataLoader(
        dataset=d, batch_sampler=batch_sampler, collate_fn=collator, return_list=True
    )
    for dd in loader():
        model_input = {
            "input_ids": dd[0],
            "tok_to_orig_start_index": dd[1],
            "tok_to_orig_end_index": dd[2],
            "labels": dd[3]
        }
        print(model_input)