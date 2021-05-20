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
"""train and evaluate"""
import tqdm
import json
import numpy as np
import os
import paddle.fluid as F
from tensorboardX import SummaryWriter


def multi_device(reader, dev_count):
    """multi device"""
    if dev_count == 1:
        for batch in reader:
            yield batch
    else:
        batches = []
        for batch in reader:
            batches.append(batch)
            if len(batches) == dev_count:
                yield batches
                batches = []


def evaluate(model, valid_exe, valid_ds, valid_prog, dev_count, metric):
    """evaluate """
    acc_loss = 0
    acc_top1 = 0
    cc = 0
    for feed_dict in tqdm.tqdm(
            multi_device(valid_ds.generator(), dev_count), desc='evaluating'):
        if dev_count > 1:
            loss, top1 = valid_exe.run(
                feed=feed_dict,
                fetch_list=[model.metrics[0].name, model.metrics[1].name])
            loss = np.mean(loss)
            top1 = np.mean(top1)
        else:
            loss, top1 = valid_exe.run(
                valid_prog,
                feed=feed_dict,
                fetch_list=[model.metrics[0].name, model.metrics[1].name])
        acc_loss += loss
        acc_top1 += top1
        cc += 1
    ret = {"loss": float(acc_loss / cc), "top1": float(acc_top1 / cc)}
    return ret


def _create_if_not_exist(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def train_and_evaluate(exe,
                       train_exe,
                       valid_exe,
                       train_ds,
                       valid_ds,
                       train_prog,
                       valid_prog,
                       model,
                       metric,
                       epoch=20,
                       dev_count=1,
                       train_log_step=5,
                       eval_step=10000,
                       output_path=None):
    """train and evaluate"""

    global_step = 0

    log_path = os.path.join(output_path, "log")
    _create_if_not_exist(log_path)

    writer = SummaryWriter(log_path)

    best_model = 0
    for e in range(epoch):
        for feed_dict in tqdm.tqdm(
                multi_device(train_ds.generator(), dev_count),
                desc='Epoch %s' % e):
            if dev_count > 1:
                ret = train_exe.run(feed=feed_dict, fetch_list=metric.vars)
                ret = [[np.mean(v)] for v in ret]
            else:
                ret = train_exe.run(train_prog,
                                    feed=feed_dict,
                                    fetch_list=metric.vars)

            ret = metric.parse(ret)
            if global_step % train_log_step == 0:
                writer.add_scalar(
                    "batch_loss", ret['loss'], global_step=global_step)
                writer.add_scalar(
                    "batch_top1", ret['top1'], global_step=global_step)

            global_step += 1
            if global_step % eval_step == 0:
                eval_ret = evaluate(model, exe, valid_ds, valid_prog, 1,
                                    metric)
                writer.add_scalar(
                    "eval_loss", eval_ret['loss'], global_step=global_step)
                writer.add_scalar(
                    "eval_top1", eval_ret['top1'], global_step=global_step)

                if eval_ret["top1"] > best_model:

                    model_name = output_path.split('/')[-1]
                    log_str = "Model: %s Step: %s Loss: %s Eval Top1: %s" % (model_name, global_step, eval_ret["loss"], eval_ret["top1"]) 

                    F.io.save_persistables(
                        exe,
                        os.path.join(output_path, "checkpoint"), train_prog)
                    eval_ret["step"] = global_step
                    with open(os.path.join(output_path, "best.txt"), "w") as f:
                        f.write(json.dumps(eval_ret, indent=2) + '\n')
                    best_model = eval_ret["top1"]
        # Epoch End
        eval_ret = evaluate(model, exe, valid_ds, valid_prog, 1,
            metric)
        writer.add_scalar(
            "eval_loss", eval_ret['loss'], global_step=global_step)
        writer.add_scalar(
            "eval_top1", eval_ret['top1'], global_step=global_step)

        if eval_ret["top1"] > best_model:
            model_name = output_path.split('/')[-1]
            log_str = "Model: %s Step: %s Loss: %s Eval Top1: %s" % (model_name, global_step, eval_ret["loss"], eval_ret["top1"]) 

            F.io.save_persistables(
                exe,
                os.path.join(output_path, "checkpoint"), train_prog)
            eval_ret["step"] = global_step
            with open(os.path.join(output_path, "best.txt"), "w") as f:
                f.write(json.dumps(eval_ret, indent=2) + '\n')
            best_model = eval_ret["top1"]

    writer.close()
