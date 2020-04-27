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
"""Finetuning on reading comprehension task """

import argparse
import ast
import json
import os

import paddle.fluid as fluid
import paddlehub as hub

from demo_dataset import DuReader

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=1, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Warmup proportion params for warmup strategy")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=384, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=8, help="Total examples' number in batch for training.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=True, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.


if __name__ == '__main__':
    # 加载PaddleHub ERNIE预训练模型
    module = hub.Module(name="ernie")
    
    # ERNIE预训练模型输入变量inputs、输出变量outputs、以及模型program
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len)

    # 加载竞赛数据集并使用ReadingComprehensionReader读取数据
    dataset = DuReader()
    reader = hub.reader.ReadingComprehensionReader(
        dataset=dataset,
        vocab_path=module.get_vocab_path(),
        max_seq_len=args.max_seq_len,
        doc_stride=128,
        max_query_length=64)

    # 取ERNIE的字级别预训练输出
    seq_output = outputs["sequence_output"]

    # 设置运行program所需的feed_list
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # 选择Fine-tune优化策略
    strategy = hub.AdamWeightDecayStrategy(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion)

    # 设置运行配置
    config = hub.RunConfig(
        eval_interval=500,
        use_pyreader=False,
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # 定义阅读理解Fine-tune Task
    # 由于竞赛数据集与cmrc2018数据集格式比较相似，此处sub_task应为cmrc2018
    # 否则运行可能出错
    reading_comprehension_task = hub.ReadingComprehensionTask(
        data_reader=reader,
        feature=seq_output,
        feed_list=feed_list,
        config=config,
        sub_task="cmrc2018",
    )
    
    # 调用finetune_and_eval API，将会自动进行训练、评估以及保存最佳模型
    reading_comprehension_task.finetune_and_eval()
    
    # 竞赛数据集测试集部分数据用于预测
    data = dataset.predict_examples
    # 调用predict接口, 打开return_result(True)，将自动返回预测结果
    all_prediction = reading_comprehension_task.predict(data=data, return_result=True)
    # 写入预测结果
    json.dump(all_prediction, open('submit.json', 'w'), ensure_ascii=False)
