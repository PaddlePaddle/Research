# -*- coding: utf-8 -*-
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

import argparse
import json
import logging
import multiprocessing
import os
import time

import numpy as np

os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import paddle.fluid as fluid
from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

import reader.type_pairwise_ranker_reader as task_reader
from model.ernie import ErnieConfig
from finetune.type_pairwise_ranker import create_model, evaluate, predict

from utils.args import print_arguments, check_cuda, prepare_logger, ArgumentGroup
from utils.init import init_pretraining_params
from finetune_args import parser

parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(
    parser, "model", "options to init, resume and save model.",
)
model_g.add_arg(
    "ernie_config_path", str, None,
    "Path to the json file for ernie model config.",
)
model_g.add_arg(
    "init_checkpoint", str, None,
    "Init checkpoint to resume training from.",
)
model_g.add_arg(
    "save_inference_model_path", str, "inference_model",
    "If set, save the inference model to this path.",
)
model_g.add_arg(
    "use_fp16", bool, False,
    "Whether to resume parameters from fp16 checkpoint.",
)
model_g.add_arg("num_labels", int, 2, "num labels for classify")

data_g = ArgumentGroup(
    parser, "data", "Data paths, vocab paths and data processing options.",
)
data_g.add_arg("predict_set", str, None, "Predict set file")
data_g.add_arg("vocab_path", str, None, "Vocabulary path.")
data_g.add_arg("label_map_config", str, None, "Label_map_config json file.")
data_g.add_arg(
    "max_seq_len", int, 128,
    "Number of words of the longest seqence.",
)
data_g.add_arg(
    "batch_size", int, 32,
    "Total examples' number in batch for training. see also --in_tokens.",
)
data_g.add_arg(
    "do_lower_case", bool, True,
    "Whether to lower case the input text. \
                Should be True for uncased models and False for cased models.",
)

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg(
    "do_prediction", bool, True,
    "Whether to do prediction on test set.",
)

args = parser.parse_args()
log = logging.getLogger()


def main(args):
    """main"""
    ernie_config = ErnieConfig(args.ernie_config_path)
    ernie_config.print_config()

    reader = task_reader.RankReader(
        vocab_path=args.vocab_path,
        label_map_config=args.label_map_config,
        max_seq_len=args.max_seq_len,
        do_lower_case=args.do_lower_case,
        in_tokens=False,
        is_inference=True,
    )

    predict_prog = fluid.Program()
    predict_startup = fluid.Program()
    with fluid.program_guard(predict_prog, predict_startup):
        with fluid.unique_name.guard():
            ret = create_model(
                args,
                pyreader_name='predict_reader',
                ernie_config=ernie_config,
                is_classify=True,
                is_prediction=True,
            )
            predict_pyreader = ret['pyreader']
            left_score = ret['left_probs']
            right_score = ret['right_probs']
            type_probs = ret['type_probs']
            feed_targets_name = ret['feed_targets_name']

    predict_prog = predict_prog.clone(for_test=True)

    if args.use_cuda:
        dev_list = fluid.cuda_places()
        place = dev_list[0]
        print('----------place-----------')
        print(place)
        dev_count = len(dev_list)
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    place = fluid.CUDAPlace(0) if args.use_cuda == True else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(predict_startup)

    if args.init_checkpoint:
        init_pretraining_params(exe, args.init_checkpoint, predict_prog)
    else:
        raise ValueError(
            "args 'init_checkpoint' should be set for prediction!",
        )

    assert args.save_inference_model_path, \
        "args save_inference_model_path should be set for prediction"
    _, ckpt_dir = os.path.split(args.init_checkpoint.rstrip('/'))
    dir_name = ckpt_dir + '_inference_model'
    model_path = os.path.join(args.save_inference_model_path, dir_name)
    log.info("save inference model to %s" % model_path)
    fluid.io.save_inference_model(
        model_path,
        feed_targets_name, [left_score, right_score, type_probs],
        exe,
        main_program=predict_prog,
    )

    config = AnalysisConfig(model_path)
    if not args.use_cuda:
        log.info("disable gpu")
        config.disable_gpu()
    else:
        log.info("using gpu")
        config.enable_use_gpu(1024)

    # Create PaddlePredictor
    predictor = create_paddle_predictor(config)

    predict_data_generator = reader.data_generator(
        input_file=args.predict_set,
        batch_size=args.batch_size,
        epoch=1,
        shuffle=False,
    )

    log.info("-------------- prediction results --------------")
    np.set_printoptions(precision=4, suppress=True)
    index = 0
    total_time = 0
    qid_total = None
    left_score_total = None
    right_score_total = None
    type_prob_total = None
    ent_id_total = None
    for sample in predict_data_generator():
        src_ids_1 = sample[0]
        sent_ids_1 = sample[1]
        pos_ids_1 = sample[2]
        task_ids_1 = sample[3]
        input_mask_1 = sample[4]
        src_ids_2 = sample[5]
        sent_ids_2 = sample[6]
        pos_ids_2 = sample[7]
        task_ids_2 = sample[8]
        input_mask_2 = sample[9]
        src_ids_3 = sample[10]
        sent_ids_3 = sample[11]
        pos_ids_3 = sample[12]
        task_ids_3 = sample[13]
        input_mask_3 = sample[14]
        qids = sample[15]
        ent_ids = sample[16]

        inputs = [array2tensor(ndarray) for ndarray in [
            src_ids_1, sent_ids_1, pos_ids_1, task_ids_1, input_mask_1,
            src_ids_2, sent_ids_2, pos_ids_2, task_ids_2, input_mask_2,
            src_ids_3, sent_ids_3, pos_ids_3, task_ids_3, input_mask_3,
            qids,
        ]]
        begin_time = time.time()
        outputs = predictor.run(inputs)
        end_time = time.time()
        total_time += end_time - begin_time

        output_l = outputs[0]
        output_r = outputs[1]
        output_t = outputs[2]

        output_left = output_l.data.float_data()
        output_right = output_r.data.float_data()
        output_type = output_t.data.float_data()
        output_type = np.array(output_type)

        batch_result_left = np.array(output_left).reshape(output_l.shape)
        batch_result_right = np.array(output_right).reshape(output_r.shape)
        batch_result_type = np.array(output_type).reshape(
            int(output_type.shape[0]/24), 24,
        )

        if ent_id_total is None:
            ent_id_total = ent_ids
        else:
            ent_id_total = np.concatenate((ent_id_total, ent_ids), axis=0)

        if qid_total is None:
            qid_total = qids
        else:
            qid_total = np.concatenate((qid_total, qids), axis=0)

        if left_score_total is None:
            left_score_total = batch_result_left
        else:
            left_score_total = np.concatenate(
                (left_score_total, batch_result_left), axis=0,
            )

        if right_score_total is None:
            right_score_total = batch_result_right
        else:
            right_score_total = np.concatenate(
                (right_score_total, batch_result_right), axis=0,
            )

        if type_prob_total is None:
            type_prob_total = batch_result_type
        else:
            type_prob_total = np.concatenate(
                (type_prob_total, batch_result_type), axis=0,
            )
    predict_res = {}
    predict_res['qid_total'] = qid_total
    predict_res['left_score_total'] = left_score_total
    predict_res['type_prob_total'] = type_prob_total
    predict_res['ent_id_total'] = ent_id_total
    predict_post_process(predict_res)


def predict_post_process(predict_res, type_label_map_reverse_path='./data/generated/type_label_map_reverse.json'):
    ent_type_dic = {}
    for ent_info in open('./data/basic_data/kb.json'):
        ent_info = json.loads(ent_info.strip())
        subject_id = ent_info['subject_id']
        subject_type = ent_info['type']
        ent_type_dic[subject_id] = subject_type

    type_label_map_reverse = json.load(open(type_label_map_reverse_path))
    qid_total = predict_res['qid_total']
    left_score_total = predict_res['left_score_total']
    type_prob_total = predict_res['type_prob_total']
    ent_id_total = predict_res['ent_id_total']
    qid_current = qid_total[0][0]
    left_score_qid = []
    type_qid = None
    ent_id_cand = []
    qid_pred = {}
    for qid, left, type_prob, ent_id in zip(
        qid_total, left_score_total,
        type_prob_total, ent_id_total,
    ):
        if qid[0] == qid_current:
            left_score_qid.append(left[0])
            type_qid = np.argmax(type_prob)
            ent_id_cand.append(ent_id[0])
        if qid[0] != qid_current:
            pred_type = type_label_map_reverse[str(type_qid)]
            score = []
            for i in range(len(ent_id_cand)):
                if ent_id_cand[i] == 'NIL':
                    score.append(left_score_qid[i] * 0.5 + 0.4)
                elif pred_type in ent_type_dic[ent_id_cand[i]]:
                    score.append(left_score_qid[i] * 0.5 + 0.5)
                else:
                    score.append(left_score_qid[i] * 0.5)
            pred_ent = ent_id_cand[score.index(max(score))]
            if pred_ent == 'NIL':
                pred_ent = 'NIL_' + pred_type
            qid_pred[qid_current] = pred_ent
            left_score_qid = [left[0]]
            qid_current = qid[0]
            ent_id_cand = [ent_id[0]]
            type_qid = np.argmax(type_prob)
    qid_pred[qid_current] = pred_ent

    qid = 1
    outfile = open('./data/generated/test_pred.json', 'wb')
    for line in open('./data/basic_data/test.json'):
        line_json = json.loads(line.strip())
        mention_data = line_json.get('mention_data')
        mention_data_pred = []
        for item in mention_data:
            kb_id = qid_pred[qid]
            item['kb_id'] = kb_id
            mention_data_pred.append(item)
            qid += 1
        line_json['mention_data'] = mention_data_pred
        outfile.write(json.dumps(line_json, ensure_ascii=False).encode('utf8'))
        outfile.write('\n')


def array2tensor(ndarray):
    """ convert numpy array to PaddleTensor"""
    assert isinstance(ndarray, np.ndarray), "input type must be np.ndarray"
    tensor = PaddleTensor()
    tensor.name = "data"
    tensor.shape = ndarray.shape
    if "float" in str(ndarray.dtype):
        tensor.dtype = PaddleDType.FLOAT32
    elif "int" in str(ndarray.dtype):
        tensor.dtype = PaddleDType.INT64
    else:
        raise ValueError("{} type ndarray is unsupported".format(tensor.dtype))

    tensor.data = PaddleBuf(ndarray.flatten().tolist())
    return tensor


if __name__ == '__main__':
    prepare_logger(log)
    print_arguments(args)
    main(args)
