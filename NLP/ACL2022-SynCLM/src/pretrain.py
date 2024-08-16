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
"""SynCLM pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import multiprocessing

import numpy as np
import paddle.fluid as fluid

from reader.pretraining_reader import SynCLMDataReader
from model.synclm import SynCLMModel, SynCLMConfig
from utils.optimization import optimization
from utils.args import print_arguments
from utils.utils import visualdl_log
from utils.init import init_checkpoint, init_pretraining_params
from model.roberta_tokenization import GptBpeTokenizer
from args.pretrain_args import parser
from collections import OrderedDict
import paddle

paddle.enable_static()

args = parser.parse_args()


def get_time():
    res = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return res


# yapf: enable.
def create_model(pyreader_name, synclm_config):
    """create synclm model"""
    input_mask_shape = [-1, args.max_seq_len, args.max_seq_len]
    att_layers = list(map(int, args.att_layer.split(",")))

    pyreader = fluid.layers.py_reader(
        capacity=70,
        shapes=[
            [-1, args.max_seq_len, 1],  # src_ids
            [-1, args.max_seq_len, 1],  # pos_ids
            [-1, args.max_seq_len, 1],  # sent_ids
            input_mask_shape,  # input_mask
            [-1, 1],  # mask_label
            [-1, 1],  # mask_pos
            [-1, 1],  # phrase_samples
            [-1, 1],  # phrase_positives
            [-1, args.phrase_max_neg_num, 1],  # phrase_negatives
            [-1, args.phrase_max_neg_num],  # phrase_negatives_mask
            [-1, 1],  # tree_samples
            [-1, args.tree_max_sub_num, 1],  # tree_positives
            [-1, args.tree_max_sub_num],  # tree_positives_mask
            [-1, args.tree_max_neg_num, args.tree_max_sub_num + 1, 1],  # tree_negatives
            [-1, args.tree_max_neg_num, args.tree_max_sub_num + 1]  # tree_negatives_mask
        ],
        dtypes=[
            'int64', 'int64', 'int64', 'float32', 'int64', 'int64', 'int64', 'int64', 'int64', 'float32', 'int64',
            'int64', 'float32', 'int64', 'float32'
        ],
        lod_levels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        name=pyreader_name,
        use_double_buffer=True)

    (src_ids, pos_ids, sent_ids, input_mask, mask_label, mask_pos, phrase_samples, phrase_positives, phrase_negatives,
     phrase_negatives_mask, tree_samples, tree_positives, tree_positives_mask, tree_negatives,
     tree_negatives_mask) = fluid.layers.read_file(pyreader)
    emb_ids = {"word_embedding": src_ids, "sent_embedding": sent_ids, "pos_embedding": pos_ids}

    synclm = SynCLMModel(emb_ids=emb_ids,
                         input_mask=input_mask,
                         config=synclm_config,
                         weight_sharing=args.weight_sharing,
                         att_layers=att_layers)

    reshaped_emb_out, mask_lm_loss = synclm.get_mask_loss(mask_label, mask_pos)
    phrase_loss, phrase_tau = synclm.get_phrase_loss(phrase_samples, phrase_positives, phrase_negatives,
                                                     phrase_negatives_mask, input_mask, args.phrase_max_neg_num)

    tree_loss, tree_tau = synclm.get_tree_loss(reshaped_emb_out, tree_samples, tree_positives, tree_positives_mask,
                                               tree_negatives, tree_negatives_mask, input_mask, args.tree_max_neg_num,
                                               args.tree_max_sub_num)

    total_loss = mask_lm_loss + phrase_loss + tree_loss
    return pyreader, mask_lm_loss, phrase_loss, tree_loss, total_loss, phrase_tau, tree_tau


def predict_wrapper(args, exe, synclm_config, test_prog=None, pyreader=None, fetch_list=None):
    # Context to do validation.
    filelist = args.test_filelist if args.do_test else args.valid_filelist
    """load vocabulary"""
    tokenizer = GptBpeTokenizer(vocab_file=args.roberta_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=True)

    data_reader = SynCLMDataReader(filelist,
                                   tokenizer=tokenizer,
                                   in_tokens=args.in_tokens,
                                   batch_size=args.batch_size,
                                   voc_size=synclm_config['vocab_size'],
                                   shuffle_files=False,
                                   epoch=1,
                                   max_seq_len=args.max_seq_len,
                                   is_test=True,
                                   tree_max_sub_num=args.tree_max_sub_num,
                                   tree_max_neg_num=args.tree_max_neg_num,
                                   phrase_max_neg_num=args.phrase_max_neg_num)

    if args.do_test:
        assert args.init_checkpoint is not None, "[FATAL] Please use --init_checkpoint '/path/to/checkpoints' \
                                                  to specify you pretrained model checkpoints"

        init_pretraining_params(exe, args.init_checkpoint, test_prog)

    def predict(exe=exe, pyreader=pyreader):

        pyreader.decorate_tensor_provider(data_reader.data_generator())
        pyreader.start()

        lm_cost = []
        phrase_cost = []
        tree_cost = []
        total_cost = []
        steps = 0
        time_begin = time.time()
        while True:
            try:
                each_mask_lm_cost, each_phrase_cost, each_tree_cost, each_total_cost = exe.run(fetch_list=fetch_list,
                                                                                               program=test_prog)
                lm_cost.append(each_mask_lm_cost)
                phrase_cost.append(each_phrase_cost)
                tree_cost.append(each_tree_cost)
                total_cost.append(each_total_cost)

                steps += 1
                if args.do_test and steps % args.skip_steps == 0:
                    print("[test_set] steps: %d" % steps)

            except fluid.core.EOFException:
                pyreader.reset()
                break

        used_time = time.time() - time_begin
        return np.sum(np.array(lm_cost)), np.sum(np.array(phrase_cost)), np.sum(np.array(tree_cost)), np.sum(
            np.array(total_cost)), steps, (args.skip_steps / used_time)

    return predict


def train(args):
    synclm_config = SynCLMConfig(args.synclm_config_path)
    synclm_config.print_config()
    print("pretraining start")

    node_nums = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
    train_program = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_program, startup_prog):
        with fluid.unique_name.guard():
            train_pyreader, mask_lm_loss, phrase_loss, tree_loss, total_loss, tau_term, tau_con = create_model(
                pyreader_name='train_reader', synclm_config=synclm_config)
            scheduled_lr, loss_scaling = optimization(loss=total_loss,
                                                      warmup_steps=args.warmup_steps,
                                                      num_train_steps=args.num_train_steps,
                                                      learning_rate=args.learning_rate,
                                                      train_program=train_program,
                                                      weight_decay=args.weight_decay,
                                                      scheduler=args.lr_scheduler,
                                                      use_fp16=args.use_fp16,
                                                      use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                                                      init_loss_scaling=args.init_loss_scaling,
                                                      beta1=args.beta1,
                                                      beta2=args.beta2,
                                                      epsilon=args.epsilon)

    test_prog = fluid.Program()
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            test_pyreader, mask_lm_loss, phrase_loss, tree_loss, total_loss, tau_term, tau_con = create_model(
                pyreader_name='test_reader', synclm_config=synclm_config)

    test_prog = test_prog.clone(for_test=True)

    gpu_id = 0
    gpus = fluid.core.get_cuda_device_count()
    if args.is_distributed and os.getenv("FLAGS_selected_gpus") is not None:
        gpu_list = os.getenv("FLAGS_selected_gpus").split(",")
        gpus = len(gpu_list)
        gpu_id = int(gpu_list[0])

    if args.use_cuda:
        place = fluid.CUDAPlace(gpu_id)
        dev_count = gpus
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    print("Device count %d, gpu_id:%d" % (dev_count, gpu_id))
    print("theoretical memory usage: ")
    if args.in_tokens:
        print(fluid.contrib.memory_usage(program=train_program, batch_size=args.batch_size // args.max_seq_len))

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    print("args.is_distributed:", args.is_distributed)
    if args.is_distributed:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        worker_endpoints = worker_endpoints_env.split(",")
        trainers_num = len(worker_endpoints)

        print("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
              trainer_id:{}".format(worker_endpoints, trainers_num, current_endpoint, trainer_id))
        # prepare nccl2 env.
        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        if args.nccl_comm_num > 1:
            config.nccl_comm_num = args.nccl_comm_num
        if args.use_hierarchical_allreduce and trainers_num > args.hierarchical_allreduce_inter_nranks:
            config.use_hierarchical_allreduce = args.use_hierarchical_allreduce
            config.hierarchical_allreduce_inter_nranks = args.hierarchical_allreduce_inter_nranks

            assert config.hierarchical_allreduce_inter_nranks > 1
            assert trainers_num % config.hierarchical_allreduce_inter_nranks == 0

            config.hierarchical_allreduce_exter_nranks = \
                trainers_num / config.hierarchical_allreduce_inter_nranks

        t = fluid.DistributeTranspiler(config=config)
        t.transpile(trainer_id,
                    trainers=worker_endpoints_env,
                    current_endpoint=current_endpoint,
                    program=train_program,
                    startup_program=startup_prog)
        nccl2_num_trainers = trainers_num
        nccl2_trainer_id = trainer_id

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.init_checkpoint and args.init_checkpoint != "":
        init_checkpoint(exe, args.init_checkpoint, train_program)
    """load vocabulary"""
    tokenizer = GptBpeTokenizer(vocab_file=args.roberta_vocab_file,
                                encoder_json_file=args.encoder_json_file,
                                vocab_bpe_file=args.vocab_bpe_file,
                                do_lower_case=True)

    data_reader = SynCLMDataReader(filelist=args.train_filelist,
                                   tokenizer=tokenizer,
                                   in_tokens=args.in_tokens,
                                   batch_size=args.batch_size,
                                   voc_size=synclm_config['vocab_size'],
                                   random_seed=args.random_seed,
                                   epoch=args.epoch,
                                   max_seq_len=args.max_seq_len,
                                   tree_max_sub_num=args.tree_max_sub_num,
                                   tree_max_neg_num=args.tree_max_neg_num,
                                   phrase_max_neg_num=args.phrase_max_neg_num)

    exec_strategy = fluid.ExecutionStrategy()
    if args.use_fast_executor:
        exec_strategy.use_experimental_executor = True
    exec_strategy.num_threads = 4 if args.use_fp16 else 2  # 2 for fp32 4 for fp16
    exec_strategy.num_iteration_per_drop_scope = min(args.num_iteration_per_drop_scope, args.skip_steps)

    build_strategy = fluid.BuildStrategy()
    build_strategy.remove_unnecessary_lock = False

    if args.use_fuse:
        build_strategy.fuse_all_reduce_ops = True

    train_exe = fluid.ParallelExecutor(use_cuda=args.use_cuda,
                                       loss_name=total_loss.name,
                                       build_strategy=build_strategy,
                                       exec_strategy=exec_strategy,
                                       main_program=train_program,
                                       num_trainers=nccl2_num_trainers,
                                       trainer_id=nccl2_trainer_id)

    if args.valid_filelist and args.valid_filelist != "":
        predict = predict_wrapper(args,
                                  exe,
                                  synclm_config,
                                  test_prog=test_prog,
                                  pyreader=test_pyreader,
                                  fetch_list=[mask_lm_loss.name, phrase_loss.name, tree_loss.name, total_loss.name])

    train_pyreader.decorate_tensor_provider(data_reader.data_generator())
    train_pyreader.start()
    steps = args.init_step
    lm_cost = []
    phrase_cost = []
    tree_cost = []
    total_cost = []

    tau_terms = []
    tau_cons = []
    time_begin = time.time()
    if args.valid_filelist:
        vali_mlm_cost, vali_phrase_cost, vali_tree_cost, vali_total_cost, vali_steps, vali_speed = predict()
        vali_mlm_cost = np.mean(np.array(vali_mlm_cost) / vali_steps)
        vali_phrase_cost = np.mean(np.array(vali_phrase_cost) / vali_steps)
        vali_tree_cost = np.mean(np.array(vali_tree_cost) / vali_steps)

        vali_total_cost = np.mean(np.array(vali_total_cost) / vali_steps)
        ppl = np.exp(np.mean(np.array(vali_mlm_cost) / vali_steps))
        print("[validation_set] %s - "
              "mlm_loss: %f, phrase_loss: %f,  tree_loss: %f,  total_loss: %f, global ppl: %f, speed: %f steps/s" %
              (get_time(), vali_mlm_cost, vali_phrase_cost, vali_tree_cost, vali_total_cost, ppl, vali_speed))
    print("start training...")
    while steps < args.num_train_steps:
        try:
            steps += 1
            skip_steps = args.skip_steps

            fetch_list = []
            if nccl2_trainer_id == 0 and steps % skip_steps == 0:
                fetch_list = [
                    mask_lm_loss.name, phrase_loss.name, tree_loss.name, total_loss.name, scheduled_lr.name,
                    tau_term.name, tau_con.name
                ]
                if args.use_fp16:
                    fetch_list.append(loss_scaling.name)

            ret = train_exe.run(fetch_list=fetch_list)
            time_end = time.time()
            used_time = time_end - time_begin

            if ret:
                each_mask_lm_cost, each_phrase_loss, each_tree_loss, each_total_loss, np_lr, tau_t, tau_c, l_scaling = ret if args.use_fp16 else ret + [
                    [args.init_loss_scaling]
                ]
                lm_cost.extend(each_mask_lm_cost)
                phrase_cost.extend(each_phrase_loss)
                tree_cost.extend(each_tree_loss)
                total_cost.extend(each_total_loss)

                tau_terms.extend(tau_t)
                tau_cons.extend(tau_c)

                epoch, current_file_index, total_file, current_file = data_reader.get_progress()

                print("feed_queue size", train_pyreader.queue.size())
                print("current learning_rate:%.8f, loss scaling:%f" % (np_lr[0], l_scaling[0]))
                lm_cost = np.mean(np.array(lm_cost))
                phrase_cost = np.mean(np.array(phrase_cost))
                tree_cost = np.mean(np.array(tree_cost))
                total_cost = np.mean(np.array(total_cost))
                tau_terms = np.mean(np.array(tau_terms))
                tau_cons = np.mean(np.array(tau_cons))

                ppl = np.exp(np.mean(np.array(lm_cost)))
                print(
                    "%s - epoch: %d, progress: %d/%d, step: %d, mlm_loss: %f,   phrase_loss: %f,   tree_loss: %f,   total_loss: %f, ppl: %f, tau_terms: %f, tau_cons: %f, speed: %f steps/s, file: %s"
                    % (get_time(), epoch, current_file_index, total_file, steps, lm_cost, phrase_cost, tree_cost,
                       total_cost, ppl, tau_terms, tau_cons, skip_steps / used_time, current_file))
                if args.visualdl_log:
                    visuallog_dict = OrderedDict()
                    visuallog_dict["ppl"] = ppl
                    visuallog_dict["lm_cost"] = lm_cost
                    visuallog_dict["phrase_cost"] = phrase_cost
                    visuallog_dict["tree_cost"] = tree_cost
                    visuallog_dict["total_cost"] = total_cost
                    visualdl_log(visuallog_dict, lm_cost, phrase_cost, tree_cost, total_cost, steps, phase='train')

                lm_cost = []
                phrase_cost = []
                tree_cost = []
                total_cost = []
                tau_terms = []
                tau_cons = []
                time_begin = time.time()
            elif steps % skip_steps == 0:
                epoch, current_file_index, total_file, current_file = data_reader.get_progress()
                print("feed_queue size", train_pyreader.queue.size())
                print("%s - epoch: %d, progress: %d/%d, step: %d, "
                      "speed: %f steps/s, file: %s" %
                      (get_time(), epoch, current_file_index, total_file, steps, skip_steps / used_time, current_file))
                time_begin = time.time()

            if not nccl2_trainer_id == 0:
                continue

            if steps % args.save_steps == 0:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)

            if args.valid_filelist and steps % args.validation_steps == 0:
                vali_mlm_cost, vali_phrase_cost, vali_tree_cost, vali_total_cost, vali_steps, vali_speed = predict()
                vali_mlm_cost = np.mean(np.array(vali_mlm_cost) / vali_steps)
                vali_phrase_cost = np.mean(np.array(vali_phrase_cost) / vali_steps)
                vali_tree_cost = np.mean(np.array(vali_tree_cost) / vali_steps)

                vali_total_cost = np.mean(np.array(vali_total_cost) / vali_steps)
                ppl = np.exp(np.mean(np.array(vali_mlm_cost) / vali_steps))
                print(
                    "[validation_set] %s - epoch: %d, step: %d, "
                    "mlm_loss: %f,  phrase_loss: %f, tree_loss: %f,  total_loss: %f, global ppl: %f, speed: %f steps/s"
                    % (get_time(), epoch, steps, vali_mlm_cost, vali_phrase_cost, vali_tree_cost, vali_total_cost, ppl,
                       vali_speed))

                if args.visualdl_log:
                    visuallog_dict = OrderedDict()
                    visuallog_dict["ppl"] = ppl
                    visuallog_dict["vali_mlm_cost"] = vali_mlm_cost
                    visuallog_dict["vali_phrase_cost"] = vali_phrase_cost
                    visuallog_dict["vali_tree_cost"] = vali_tree_cost
                    visuallog_dict["vali_total_cost"] = vali_total_cost
                    visualdl_log(visuallog_dict,
                                 vali_mlm_cost,
                                 vali_phrase_cost,
                                 vali_tree_cost,
                                 vali_total_cost,
                                 steps,
                                 phase='valid')

        except fluid.core.EOFException:
            train_pyreader.reset()
            break

    train_pyreader.reset()


if __name__ == '__main__':
    print_arguments(args)
    train(args)
