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
"""Finetuning on classification tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys
import time
import six
import json
import logging
import multiprocessing
from io import open

import paddle.fluid as fluid

import reader.task_reader as task_reader
from model.ernie import ErnieConfig
from optimization import optimization
from utils.init import init_pretraining_params
from utils.init import init_checkpoint
from utils.args import print_arguments
from utils.args import check_cuda
from utils.args import prepare_logger
from finetune.sequence_label import create_model
from finetune.sequence_label import evaluate
from finetune.sequence_label import predict
from finetune.sequence_label import calculate_f1
from finetune_args import parser
from utils import utils

reload(sys)
sys.setdefaultencoding("utf-8")

log = logging.getLogger()
args = parser.parse_args()
labels_map = {}  # label
for line in utils.read_by_lines(args.label_map_config):
    arr = line.split("\t")
    labels_map[arr[0]] = int(arr[1])
args.num_labels = len(labels_map)

print("=========ERNIE CONFIG============")
ernie_config = ErnieConfig(args.ernie_config_path)
ernie_config.print_config()
print("=========ERNIE CONFIG============")

if args.use_cuda:
    dev_list = fluid.cuda_places()
    place = dev_list[0]
    dev_count = len(dev_list)
else:
    place = fluid.CPUPlace()
    dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))


def main(args):
    """main"""
    reader = task_reader.RoleSequenceLabelReader(
        vocab_path=args.vocab_path,
        labels_map=labels_map,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed,
        task_id=args.task_id)

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        train_data_generator = reader.data_generator(
            input_file=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        if args.in_tokens:
            if args.batch_size < args.max_seq_len:
                raise ValueError(
                    'if in_tokens=True, batch_size should greater than max_sqelen, got batch_size:%d seqlen:%d'
                    % (args.batch_size, args.max_seq_len))

            max_train_steps = args.epoch * num_train_examples // (
                args.batch_size // args.max_seq_len) // dev_count
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // dev_count

        warmup_steps = int(max_train_steps * args.warmup_proportion)
        print("Device count: %d" % dev_count)
        print("Num train examples: %d" % num_train_examples)
        print("Max train steps: %d" % max_train_steps)
        print("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()

        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='train_reader',
                    ernie_config=ernie_config)
                scheduled_lr, loss_scaling = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                    init_loss_scaling=args.init_loss_scaling,
                    incr_every_n_steps=args.incr_every_n_steps,
                    decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
                    incr_ratio=args.incr_ratio,
                    decr_ratio=args.decr_ratio)

        if args.verbose:
            if args.in_tokens:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program,
                    batch_size=args.batch_size // args.max_seq_len)
            else:
                lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
                    program=train_program, batch_size=args.batch_size)
            print("Theoretical memory usage in training: %.3f - %.3f %s" %
                  (lower_mem, upper_mem, unit))

    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, graph_vars = create_model(
                    args,
                    pyreader_name='test_reader',
                    ernie_config=ernie_config)

        test_prog = test_prog.clone(for_test=True)

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.do_train:
        if args.init_checkpoint and args.init_pretraining_params:
            print(
                "WARNING: args 'init_checkpoint' and 'init_pretraining_params' "
                "both are set! Only arg 'init_checkpoint' is made valid.")
        if args.init_checkpoint:
            init_checkpoint(
                exe,
                args.init_checkpoint,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
        elif args.init_pretraining_params:
            init_pretraining_params(
                exe,
                args.init_pretraining_params,
                main_program=startup_prog,
                use_fp16=args.use_fp16)
    elif args.do_val or args.do_test:
        if not args.init_checkpoint:
            raise ValueError("args 'init_checkpoint' should be set if"
                             "only doing validation or testing!")
        init_checkpoint(
            exe,
            args.init_checkpoint,
            main_program=startup_prog,
            use_fp16=args.use_fp16)

    if args.do_train:
        exec_strategy = fluid.ExecutionStrategy()
        if args.use_fast_executor:
            exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = dev_count
        exec_strategy.num_iteration_per_drop_scope = args.num_iteration_per_drop_scope

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            loss_name=graph_vars["loss"].name,
            exec_strategy=exec_strategy,
            main_program=train_program,
            num_trainers=nccl2_num_trainers,
            trainer_id=nccl2_trainer_id)

        train_pyreader.decorate_tensor_provider(train_data_generator)
    else:
        train_exe = None

    if args.do_val or args.do_test:
        test_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            main_program=test_prog,
            share_vars_from=train_exe)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        graph_vars["learning_rate"] = scheduled_lr

        time_begin = time.time()
        while True:
            try:
                steps += 1
                if steps % args.skip_steps != 0:
                    train_exe.run(fetch_list=[])
                else:
                    fetch_list = [
                        graph_vars["num_infer"].name,
                        graph_vars["num_label"].name,
                        graph_vars["num_correct"].name,
                        graph_vars["loss"].name,
                        graph_vars['learning_rate'].name,
                    ]

                    out = train_exe.run(fetch_list=fetch_list)
                    num_infer, num_label, num_correct, np_loss, np_lr = out
                    lr = float(np_lr[0])
                    loss = np_loss.mean()
                    precision, recall, f1 = calculate_f1(num_label, num_infer,
                                                         num_correct)
                    if args.verbose:
                        print(
                            "train pyreader queue size: %d, learning rate: %f"
                            % (train_pyreader.queue.size(), lr
                               if warmup_steps > 0 else args.learning_rate))

                    current_example, current_epoch = reader.get_train_progress(
                    )
                    time_end = time.time()
                    used_time = time_end - time_begin
                    print(
                        u"【train】epoch: {}, step: {}, loss: {:.6f}, "
                        "f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, speed: {:.3f} steps/s"
                        .format(current_epoch, steps,
                                float(loss),
                                float(f1),
                                float(precision),
                                float(recall), args.skip_steps / used_time))
                    time_begin = time.time()

                if steps % args.save_steps == 0:
                    save_path = os.path.join(args.checkpoints,
                                             "step_" + str(steps))
                    fluid.io.save_persistables(exe, save_path, train_program)

                if steps % args.validation_steps == 0:
                    # evaluate dev set
                    if args.do_val:
                        precision, recall, f1 = evaluate_wrapper(
                            reader, exe, test_prog, test_pyreader, graph_vars,
                            current_epoch, steps)
                        print(
                            u"【dev】precision {:.4f} , recall {:.4f}, f1-score {:.4f}"
                            .format(
                                float(precision), float(recall), float(f1)))
                    # evaluate test set
                    if args.do_test:
                        precision, recall, f1 = evaluate_wrapper(
                            reader, exe, test_prog, test_pyreader, graph_vars,
                            current_epoch, steps)
                        print(
                            u"【test】precision {:.4f} , recall {:.4f}, f1-score {:.4f}"
                            .format(
                                float(precision), float(recall), float(f1)))

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "final_model")
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    # final eval on dev set
    if args.do_val:
        precision, recall, f1 = evaluate_wrapper(
            reader, exe, test_prog, test_pyreader, graph_vars, 1, 'final')
        print(u"【dev】precision {:.4f} , recall {:.4f}, f1-score {:.4f}".format(
            float(precision), float(recall), float(f1)))

    if args.do_test:
        test_ret = predict_wrapper(reader, exe, test_prog, test_pyreader,
                                   graph_vars, 1, 'final')
        utils.write_by_lines(args.trigger_pred_save_path, test_ret)


def evaluate_wrapper(reader, exe, test_prog, test_pyreader, graph_vars, epoch,
                     steps):
    """evaluate_wrapper"""
    # evaluate dev set
    batch_size = args.batch_size if args.predict_batch_size is None else args.predict_batch_size
    test_pyreader.decorate_tensor_provider(
        reader.data_generator(
            args.dev_set,
            batch_size=batch_size,
            epoch=1,
            dev_count=1,
            shuffle=False))
    precision, recall, f1 = evaluate(exe, test_prog, test_pyreader, graph_vars,
                                     args.num_labels)
    return precision, recall, f1


def predict_wrapper(reader, exe, test_prog, test_pyreader, graph_vars, epoch,
                    steps):
    """predict_wrapper"""

    def label_pred_2_ori(pred_label, ori_2_new_index):
        """label_pred_2_ori"""
        new_label = [u"O"] * len(ori_2_new_index)
        new_index = []
        for k, v in ori_2_new_index.items():
            if v == -1:
                new_index.append(k)
            elif v < len(pred_label):
                new_label[k] = pred_label[v]
        for index in new_index:
            if index == 0 or new_label[index - 1] == u"O" or index == (
                    len(new_label) - 1):
                new_label[index] = u"O"
            else:
                if new_label[index + 1] == u"O":
                    new_label[index] = u"O"
                else:
                    new_label[index] = u"I-{}".format(new_label[index - 1][2:])
        return new_label

    def get_pred_text(tokens, labels):
        """get_pred_text"""
        start, end, role_type = -1, -1, u""
        ret = []
        for i, lb in enumerate(labels):
            if lb == u"O" and start == -1 and end == -1:
                continue
            elif lb == u"O" and start > -1 and end > -1:
                ret.append({
                    "role_type": role_type,
                    "start": start,
                    "end": end,
                    "text": u"".join(tokens[start:end + 1])
                })
                start, end, role_type = -1, -1, u""
            else:
                if start == -1:
                    start, end, role_type = i, i, lb[2:]
                elif lb.startswith(u"B-"):
                    if start > -1 and end > -1:
                        ret.append({
                            "role_type": role_type,
                            "start": start,
                            "end": end,
                            "text": u"".join(tokens[start:end + 1])
                        })
                        start, end, role_type = i, i, lb[2:]
                    else:
                        start, end, role_type = i, i, lb[2:]
                elif lb[2:] == role_type:
                    end = i
                else:
                    ret.append({
                        "role_type": role_type,
                        "start": start,
                        "end": end,
                        "text": u"".join(tokens[start:end + 1])
                    })
                    start, end, role_type = i, i, lb[2:]

        if start >= 0 and end >= 0:
            ret.append({
                "role_type": role_type,
                "start": start,
                "end": end,
                "text": u"".join(tokens[start:end + 1])
            })
        return ret

    batch_size = args.batch_size if args.predict_batch_size is None else args.predict_batch_size
    test_pyreader.decorate_tensor_provider(
        reader.data_generator(
            args.test_set,
            batch_size=batch_size,
            epoch=1,
            dev_count=1,
            shuffle=False))

    examples = reader.get_examples_by_file(args.test_set)

    res = predict(exe, test_prog, test_pyreader, graph_vars, dev_count=1)
    tokenizer = reader.tokenizer
    rev_label_map = {v: k for k, v in six.iteritems(reader.label_map)}
    output = []
    print(u"examples {} res {}".format(len(examples), len(res)))

    for example, r in zip(examples, res):
        _id, s = r
        pred_tokens = tokenizer.convert_ids_to_tokens(_id)
        pred_label = [rev_label_map[ss] for ss in s]
        new_label = label_pred_2_ori(pred_label, example.ori_2_new_index)
        pred_ret = get_pred_text(pred_tokens, pred_label)
        pred_2_new_ret = get_pred_text(example.ori_text, new_label)
        output.append(
            json.dumps(
                {
                    "event_id": example.id,
                    "pred_tokens": pred_tokens,
                    "pred_labels": pred_label,
                    "tokens": example.ori_text,
                    "labels": new_label,
                    "sentence": example.sentence,
                    "pred_roles_ret": pred_ret,
                    "roles_ret": pred_2_new_ret
                },
                ensure_ascii=False))
    return output


if __name__ == '__main__':
    prepare_logger(log)
    print_arguments(args)
    check_cuda(args.use_cuda)
    main(args)
