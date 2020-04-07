#!/usr/bin/env python
#coding:utf-8
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
"""Finetuning on sequence labeling task."""

import argparse
import ast
import json
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset

from data_process import data_process
from data_process import schema_process
from data_process import write_by_lines

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--data_dir", type=str, default=None, help="data save dir")
parser.add_argument("--schema_path", type=str, default=None, help="schema path")
parser.add_argument("--train_data", type=str, default=None, help="train data")
parser.add_argument("--dev_data", type=str, default=None, help="dev data")
parser.add_argument("--test_data", type=str, default=None, help="test data")
parser.add_argument("--predict_data", type=str, default=None, help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=False, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--do_model", type=str, default="trigger", choices=["trigger", "role"], help="trigger or role")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--eval_step", type=int, default=200, help="eval step")
parser.add_argument("--model_save_step", type=int, default=3000, help="model save step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--add_crf", type=ast.literal_eval, default=True, help="add crf")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

# 先把数据处理好保存下来
train_data = data_process(args.train_data, args.do_model)  # 处理训练数据
dev_data = data_process(args.dev_data, args.do_model) # 处理dev数据
test_data = data_process(args.test_data, args.do_model)
predict_sents, predict_data = data_process(args.predict_data, args.do_model, is_predict=True)

write_by_lines("{}/{}_train.tsv".format(args.data_dir, args.do_model), train_data)
write_by_lines("{}/{}_dev.tsv".format(args.data_dir, args.do_model), dev_data)
write_by_lines("{}/{}_test.tsv".format(args.data_dir, args.do_model), test_data)
write_by_lines("{}/{}_predict.tsv".format(args.data_dir, args.do_model), predict_data)

schema_labels = schema_process(args.schema_path, args.do_model)

class EEDataset(BaseNLPDataset):
    """EEDataset"""
    def __init__(self, data_dir, labels, model="trigger"):
        # 数据集存放位置
        super(EEDataset, self).__init__(
            base_path=data_dir,
            train_file="{}_train.tsv".format(model),
            dev_file="{}_dev.tsv".format(model),
            test_file="{}_test.tsv".format(model),
            # 如果还有预测数据（不需要文本类别label），可以放在predict.tsv
            predict_file="{}_predict.tsv".format(model),
            train_file_with_header=True,
            dev_file_with_header=True,
            test_file_with_header=True,
            predict_file_with_header=True,
            # 数据集类别集合
            label_list=labels)


def main():
    # Load Paddlehub pretrained model
    # 更多预训练模型 https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel
    #model_name = "ernie_tiny"
    model_name = "chinese-roberta-wwm-ext-large"
    module = hub.Module(name=model_name)
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # Download dataset and use SequenceLabelReader to read dataset
    dataset = EEDataset(args.data_dir, schema_labels, model=args.do_model)
    reader = hub.reader.SequenceLabelReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        sp_model_path=module.get_spm_path(),
        word_dict_path=module.get_word_dict_path())

    # Construct transfer learning network
    # Use "sequence_output" for token-level output.
    sequence_output = outputs["sequence_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    feed_list = [
        inputs["input_ids"].name, inputs["position_ids"].name,
        inputs["segment_ids"].name, inputs["input_mask"].name
    ]

    # Select a finetune strategy
    strategy = hub.AdamWeightDecayStrategy(
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate)

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
            eval_interval=args.eval_step,
            save_ckpt_interval=args.model_save_step,
            use_data_parallel=args.use_data_parallel,
            use_cuda=args.use_gpu,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            strategy=strategy)

    # Define a sequence labeling finetune task by PaddleHub's API
    # If add crf, the network use crf as decoder
    seq_label_task = hub.SequenceLabelTask(
        data_reader=reader,
        feature=sequence_output,
        feed_list=feed_list,
        max_seq_len=args.max_seq_len,
        num_classes=dataset.num_labels,
        config=config,
        add_crf=args.add_crf)

    # Finetune and evaluate model by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    if args.do_train:
        print("start finetune and eval process")
        seq_label_task.finetune_and_eval()

    if args.do_predict:
        print("start predict process")
        ret = []
        id2label = {val: key for key, val in reader.label_map.items()}
        input_data = [[d] for d in predict_data]
        run_states = seq_label_task.predict(data=input_data[1:])
        results = []
        for batch_states in run_states:
            batch_results = batch_states.run_results
            batch_infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
            seq_lens = batch_results[1].reshape([-1]).astype(np.int32).tolist()
            current_id = 0
            for length in seq_lens:
                seq_infers = batch_infers[current_id:current_id + length]
                seq_result = list(map(id2label.get, seq_infers[1: -1]))
                current_id += length if args.add_crf else args.max_seq_len
                results.append(seq_result)

        ret = []
        for sent, r_label in zip(predict_sents, results):
            sent["labels"] = r_label
            ret.append(json.dumps(sent, ensure_ascii=False))
        write_by_lines("{}.{}.pred".format(args.predict_data, args.do_model), ret)


if __name__ == "__main__":
    main()
