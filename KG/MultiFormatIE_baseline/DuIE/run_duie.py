# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import random
import time
import math
import json
from functools import partial
import codecs
import zipfile
import re
from tqdm import tqdm
import sys

import numpy as np
import paddle
from paddle.io import DataLoader

from paddlenlp.transformers import ErnieTokenizer

from data_loader import DuIEDataset
from data_loader import DataCollator
from model import ErnieModelForDuIE

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name")
    parser.add_argument(
        "--do_train",
        action='store_true',
        default=False,
        help="do train")
    parser.add_argument(
        "--do_predict",
        action='store_true',
        default=False,
        help="do predict")
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        required=False,
        help="Path to initialize params from")
    parser.add_argument(
        "--data_path",
        default="./data",
        type=str,
        required=False,
        help="Path to data.")
    parser.add_argument(
        "--predict_data_file",
        default="./data/test_data.json",
        type=str,
        required=False,
        help="Path to data.")
    parser.add_argument(
        "--output_dir",
        default="./checkpoints",
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0,
        type=float,
        help="Linear warmup over warmup_ratio * total_steps.")

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    args = parser.parse_args()
    return args


def set_seed(args):
    """sets random seed"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def find_entity(text_raw, id_, predictions, tok_to_orig_start_index, tok_to_orig_end_index):
    """
    retrieval entity mention under given predicate id for certain prediction.
    this is called by the "decoding" func.
    """
    entity_list = []
    for i in range(len(predictions)):
        if [id_] in predictions[i]:
            j = 0
            while i + j + 1 < len(predictions):
                if [1] in predictions[i + j + 1]:
                    j += 1
                else:
                    break
            entity = ''.join(text_raw[tok_to_orig_start_index[i]:
                                    tok_to_orig_end_index[i + j] + 1])
            entity_list.append(entity)
    return list(set(entity_list))


def decoding(file_path,
             id2spo,
             logits_all,
             seq_len_all,
             tok_to_orig_start_index_all,
             tok_to_orig_end_index_all):
    """
    model output logits -> formatted spo (as in data set file)
    """
    example_all = []
    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            example_all.append(json.loads(line))

    formatted_outputs = []
    for (i, (example, logits, seq_len, tok_to_orig_start_index, tok_to_orig_end_index)) in \
            enumerate(zip(example_all, logits_all, seq_len_all, tok_to_orig_start_index_all, tok_to_orig_end_index_all)):

        logits = logits[1:seq_len + 1] # slice between [CLS] and [SEP] to get valid logits
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        tok_to_orig_start_index = tok_to_orig_start_index[1:seq_len + 1]
        tok_to_orig_end_index = tok_to_orig_end_index[1:seq_len + 1]
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # format predictions into example-style output
        formatted_instance = {}
        text_raw = example['text']
        complex_relation_label = [8, 10, 26, 32, 46]
        complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]

        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))

        # fetch all valid spo by subject id
        spo_list = []
        for id_ in subject_id_list:
            if id_ in complex_relation_affi_label:
                continue # do this in the next "else" branch
            if id_ not in complex_relation_label:
                subjects = find_entity(text_raw, id_, predictions, tok_to_orig_start_index, tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions, tok_to_orig_start_index, tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": {'@value': id2spo['object_type'][id_]},
                            'subject_type': id2spo['subject_type'][id_],
                            "object": {'@value': object_},
                            "subject": subject_
                        })
            else:
                #  traverse all complex relation and look through their corresponding affiliated objects
                subjects = find_entity(text_raw, id_, predictions, tok_to_orig_start_index, tok_to_orig_end_index)
                objects = find_entity(text_raw, id_ + 55, predictions, tok_to_orig_start_index, tok_to_orig_end_index)
                for subject_ in subjects:
                    for object_ in objects:
                        object_dict = {'@value': object_}
                        object_type_dict = {'@value': id2spo['object_type'][id_].split('_')[0]}
                        if id_ in [8, 10, 32, 46
                                ] and id_ + 1 in subject_id_list:
                            id_affi = id_ + 1
                            object_dict[id2spo['object_type'][id_affi].split('_')[1]] = find_entity(text_raw, id_affi + 55, predictions, tok_to_orig_start_index, tok_to_orig_end_index)[0]
                            object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = id2spo['object_type'][id_affi].split('_')[0]
                        elif id_ == 26:
                            for id_affi in [27, 28, 29]:
                                if id_affi in subject_id_list:
                                    object_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    find_entity(text_raw, id_affi + 55, predictions, tok_to_orig_start_index, tok_to_orig_end_index)[0]
                                    object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                    id2spo['object_type'][id_affi].split('_')[0]
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object_type": object_type_dict,
                            "subject_type": id2spo['subject_type'][id_],
                            "object": object_dict,
                            "subject": subject_
                        })

        formatted_instance['text'] = example['text']
        formatted_instance['spo_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs


def evaluate(model, data_loader, file_path, mode):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under args.data_path dir for later submission or evaluation.
    """
    model.eval()
    logits_all = None
    seq_len_all = None
    tok_to_orig_start_index_all = None
    tok_to_orig_end_index_all = None
    loss_all = 0
    eval_steps = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        eval_steps += 1
        input_ids, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        loss, logits, seq_len = model(input_ids=input_ids, labels=labels)
        loss_all += loss.numpy().item()
        if logits_all is None:
            logits_all = logits.numpy()
            seq_len_all = seq_len.numpy()
            tok_to_orig_start_index_all = tok_to_orig_start_index.numpy()
            tok_to_orig_end_index_all = tok_to_orig_end_index.numpy()
        else:
            logits_all = np.append(logits_all, logits.numpy(), axis=0)
            seq_len_all = np.append(seq_len_all, seq_len.numpy(), axis=0)
            tok_to_orig_start_index_all = np.append(tok_to_orig_start_index_all, tok_to_orig_start_index.numpy(), axis=0)
            tok_to_orig_end_index_all = np.append(tok_to_orig_end_index_all, tok_to_orig_end_index.numpy(), axis=0)
    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % (loss_avg))

    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)
    formatted_outputs = decoding(file_path, id2spo, logits_all, seq_len_all, tok_to_orig_start_index_all, tok_to_orig_end_index_all)
    if mode == "predict":
        predict_file_path = os.path.join(args.data_path, 'predictions.json')
    else:
        predict_file_path = os.path.join(args.data_path, 'predict_eval.json')

    with codecs.open(predict_file_path, 'w', 'utf-8') as f:
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')
    predict_zipfile_path = predict_file_path + '.zip'
    f = zipfile.ZipFile(predict_zipfile_path, 'w', zipfile.ZIP_DEFLATED)
    f.write(predict_file_path)
    f.close()

    if mode == "eval":
        r = os.popen('python3 ./re_official_evaluation.py --golden_file={} --predict_file={}'.format(
            file_path, predict_zipfile_path))
        result = r.read()
        r.close()
        precision = float(re.search("\"precision\", \"value\":.*?}", result).group(0).lstrip("\"precision\", \"value\":").rstrip("}"))
        recall = float(re.search("\"recall\", \"value\":.*?}", result).group(0).lstrip("\"recall\", \"value\":").rstrip("}"))
        f1 = float(re.search("\"f1-score\", \"value\":.*?}", result).group(0).lstrip("\"f1-score\", \"value\":").rstrip("}"))
        os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
        return precision, recall, f1
    elif mode != "predict":
        raise Exception("wrong mode for eval func")


def do_train(args, model):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = DuIEDataset.from_file(os.path.join(args.data_path, 'train_data.json'), tokenizer, args.max_seq_length, True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    collator = DataCollator(tokenizer=tokenizer)
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_sampler=train_batch_sampler, collate_fn=collator, return_list=True
    )

    eval_file_path = os.path.join(args.data_path, 'dev_data.json')
    test_dataset = DuIEDataset.from_file(eval_file_path, tokenizer, args.max_seq_length, True)
    test_batch_sampler = paddle.io.DistributedBatchSampler(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_sampler=test_batch_sampler, collate_fn=collator, return_list=True
    )

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    steps_by_epoch = len(train_data_loader)
    num_training_steps = args.max_steps if args.max_steps > 0 else (steps_by_epoch * args.num_train_epochs)
    num_warmup_steps = num_training_steps * args.warmup_ratio
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        args.learning_rate,
        lambda current_step, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps: float(
            current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps else max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps))))

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        print("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        model.train()
        for step, batch in enumerate(train_data_loader):
            input_ids, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
            loss, _, _ = model(input_ids=input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
            loss_item = loss.numpy().item()
            if global_step % args.logging_steps == 0:
                print(
                    "epoch: %d / %d, steps: %d / %d, loss: %f, speed: %.2f step/s"
                    % (epoch, args.num_train_epochs, step, steps_by_epoch, loss_item,
                    args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if (global_step % args.save_steps == 0) & (global_step!= 0):
                print("\n=====start evaluating ckpt of %d steps=====" % global_step)
                precision, recall, f1 = evaluate(model, test_data_loader, eval_file_path, "eval")
                print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" % (100 * precision, 100 * recall, 100 * f1))
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    print("saving checkpoing model_%d.pdparams to %s " % (global_step, args.output_dir))
                    paddle.save(model.state_dict(), os.path.join(args.output_dir, "model_%d.pdparams" % global_step))
                model.train() # back to train mode
            global_step += 1
        tic_epoch = time.time() - tic_epoch
        print("epoch time footprint: %d hour %d min %d sec" % (tic_epoch // 3600, (tic_epoch % 3600) // 60, tic_epoch % 60))
    # end time evaluation
    print("\n=====start evaluating last ckpt of %d steps=====" % global_step)
    precision, recall, f1 = evaluate(model, test_data_loader, eval_file_path, "eval")
    print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" % (100 * precision, 100 * recall, 100 * f1))
    if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
        paddle.save(model.state_dict(), os.path.join(args.output_dir, "model_%d.pdparams" % global_step))
    print("\n=====training complete=====")


def do_predict(args, model):
    paddle.set_device("gpu" if args.n_gpu else "cpu")

    tokenizer = ErnieTokenizer.from_pretrained(args.model_name_or_path)

    test_dataset = DuIEDataset.from_file(args.predict_data_file, tokenizer, args.max_seq_length, True)
    collator = DataCollator(tokenizer=tokenizer)

    test_batch_sampler = paddle.io.BatchSampler(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_sampler=test_batch_sampler, collate_fn=collator, return_list=True
    )
    if not (os.path.exists(args.init_checkpoint) and os.path.isfile(args.init_checkpoint)):
        sys.exit("wrong directory: init checkpoints {} not exist".format(args.init_checkpoint))
    state_dict = paddle.load(args.init_checkpoint)
    model.set_dict(state_dict)

    print("\n=====start predicting=====")
    _ = evaluate(model, test_data_loader, args.predict_data_file, "predict")
    print("=====predicting complete=====")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args)
    # prepare model
    label_map_path = os.path.join(args.data_path, "predicate2id.json")
    if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
        sys.exit("{} dose not exists or is not a file.".format(label_map_path))
    with open(label_map_path, 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2
    model = ErnieModelForDuIE.from_pretrained(args.model_name_or_path, num_classes=num_classes)  # 2 tags for each predicate + I tag + O tag

    if args.do_train:
        if args.n_gpu > 1:
            paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
        else:
            do_train(args, model)
    if args.do_predict:
        do_predict(args, model)