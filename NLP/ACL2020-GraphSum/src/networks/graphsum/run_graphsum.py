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

import os
import re
import time
import multiprocessing
import paddle.fluid as fluid
import paddle
import codecs
import numpy as np
import sentencepiece

from utils.logging import init_logger, logger
from utils.check import check_gpu
from models.neural_modules import position_encoding_init
from tools.cal_rouge import test_rouge, rouge_results_to_str
from utils.bleu import compute_bleu

import networks.graphsum.graphsum_reader as task_reader
from networks.graphsum.graphsum_model import GraphSumConfig, GraphSumModel
from optimize.optimization import optimization
from utils.init import init_pretraining_params, init_checkpoint
from utils.args import print_arguments
from networks.graphsum.graphsum_args import parser


def print_model_params(prog):
    with open('model_params.names', 'w') as fout:
        for _idx, param in enumerate(prog.global_block().iter_parameters()):
            fout.write("param %s %s\n" % (_idx, param.name))


def main(args):
    """Run GraphSum model."""

    model_config = GraphSumConfig(args.config_path)
    model_config.print_config()

    gpu_id = 0
    gpus = fluid.core.get_cuda_device_count()
    if args.is_distributed:
        gpus = os.getenv("FLAGS_selected_gpus").split(",")
        gpu_id = int(gpus[0])

    if args.use_cuda:
        place = fluid.CUDAPlace(gpu_id)
        dev_count = len(gpus) if args.is_distributed else gpus
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    """load vocabulary"""
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': spm.PieceToId('<PAD>'),
               'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>'),
               'UNK': spm.PieceToId('<UNK>')}
    logger.info(symbols)
    vocab_size = len(spm)

    """create transformer model"""
    graphsum = GraphSumModel(args=args, config=model_config,
                             padding_idx=symbols['PAD'],
                             bos_idx=symbols['BOS'],
                             eos_idx=symbols['EOS'],
                             tokenizer=spm)

    reader = task_reader.GraphSumReader(
        max_para_num=args.max_para_num,
        max_para_len=args.max_para_len,
        max_tgt_len=args.max_tgt_len,
        in_tokens=args.in_tokens,
        random_seed=args.random_seed,
        bos_idx=symbols['BOS'],
        eos_idx=symbols['EOS'],
        pad_idx=symbols['PAD'],
        n_head=model_config['num_attention_heads'])

    if not (args.do_train or args.do_val or args.do_test):
        raise ValueError("For args `do_train`, `do_val` and `do_test`, at "
                         "least one of them must be True.")

    startup_prog = fluid.Program()
    if args.random_seed is not None:
        startup_prog.random_seed = args.random_seed

    if args.do_train:
        trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", 1))
        train_data_generator = reader.data_generator_with_buffer(
            data_path=args.train_set,
            batch_size=args.batch_size,
            epoch=args.epoch,
            dev_count=trainers_num,
            shuffle=True,
            phase="train")

        num_train_examples = reader.get_num_examples(args.train_set)

        if args.in_tokens:
            max_train_steps = args.epoch * num_train_examples // (
                    args.batch_size // args.max_tgt_len) // trainers_num
        else:
            max_train_steps = args.epoch * num_train_examples // args.batch_size // trainers_num

        if args.lr_scheduler == 'linear_warmup_decay':
            warmup_steps = int(max_train_steps * args.warmup_proportion)
        else:
            warmup_steps = args.warmup_steps

        logger.info("Device count: %d, gpu_id: %d" % (dev_count, gpu_id))
        logger.info("Num train examples: %d" % num_train_examples)
        logger.info("Max train steps: %d" % max_train_steps)
        logger.info("Num warmup steps: %d" % warmup_steps)

        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_prog):
            with fluid.unique_name.guard():
                train_pyreader, graph_vars = graphsum.create_model(
                    pyreader_name='train_reader')
                scheduled_lr, _ = optimization(
                    loss=graph_vars["loss"],
                    warmup_steps=warmup_steps,
                    num_train_steps=max_train_steps,
                    learning_rate=args.learning_rate,
                    train_program=train_program,
                    startup_prog=startup_prog,
                    weight_decay=args.weight_decay,
                    d_model=model_config['hidden_size'],
                    scheduler=args.lr_scheduler,
                    use_fp16=args.use_fp16,
                    use_dynamic_loss_scaling=args.use_dynamic_loss_scaling,
                    init_loss_scaling=args.init_loss_scaling,
                    incr_every_n_steps=args.incr_every_n_steps,
                    decr_every_n_nan_or_inf=args.decr_every_n_nan_or_inf,
                    incr_ratio=args.incr_ratio,
                    decr_ratio=args.decr_ratio,
                    grad_norm=args.grad_norm,
                    beta1=args.beta1,
                    beta2=args.beta2,
                    epsilon=float(args.eps))
                """
                fluid.memory_optimize(
                    input_program=train_program,
                    skip_opt_set=[
                        graph_vars["loss"].name
                    ])
                """

        # if args.verbose:
        #     if args.in_tokens:
        #         lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
        #             program=train_program,
        #             batch_size=args.batch_size // args.max_tgt_len)
        #     else:
        #         lower_mem, upper_mem, unit = fluid.contrib.memory_usage(
        #             program=train_program, batch_size=args.batch_size)
        #     logger.info("Theoretical memory usage in training: %.3f - %.3f %s" %
        #                 (lower_mem, upper_mem, unit))

    if args.do_val or args.do_test:
        test_prog = fluid.Program()
        with fluid.program_guard(test_prog, startup_prog):
            with fluid.unique_name.guard():
                test_pyreader, test_graph_vars = graphsum.create_model(
                    pyreader_name='test_reader',
                    is_prediction=args.do_dec)

        test_prog = test_prog.clone(for_test=True)
        print_model_params(test_prog)

    if args.do_dec:
        if not os.path.exists(args.decode_path):
            os.mkdir(args.decode_path)

    nccl2_num_trainers = 1
    nccl2_trainer_id = 0
    logger.info("args.is_distributed: %s" % str(args.is_distributed))
    if args.is_distributed:
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        worker_endpoints_env = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        worker_endpoints = worker_endpoints_env.split(",")
        trainers_num = len(worker_endpoints)

        logger.info("worker_endpoints:{} trainers_num:{} current_endpoint:{} \
              trainer_id:{}".format(worker_endpoints, trainers_num,
                                    current_endpoint, trainer_id))

        # prepare nccl2 env.
        config = fluid.DistributeTranspilerConfig()
        config.mode = "nccl2"
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id,
            trainers=worker_endpoints_env,
            current_endpoint=current_endpoint,
            program=train_program if args.do_train else test_prog,
            startup_program=startup_prog)
        nccl2_num_trainers = trainers_num
        nccl2_trainer_id = trainer_id

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    if args.do_train:
        # init position_encoding
        enc_word_pos_emb_param = fluid.global_scope().find_var(
            model_config['enc_word_pos_embedding_name']).get_tensor()
        enc_word_pos_emb_param.set(
            position_encoding_init(model_config['max_position_embeddings'],
                                   model_config['hidden_size'] // 2), place)

        enc_sent_pos_emb_param = fluid.global_scope().find_var(
            model_config['enc_sen_pos_embedding_name']).get_tensor()
        enc_sent_pos_emb_param.set(
            position_encoding_init(model_config['max_position_embeddings'],
                                   model_config['hidden_size'] // 2), place)

        dec_word_pos_emb_param = fluid.global_scope().find_var(
            model_config['dec_word_pos_embedding_name']).get_tensor()
        dec_word_pos_emb_param.set(
            position_encoding_init(model_config['max_position_embeddings'],
                                   model_config['hidden_size']), place)

        if args.init_checkpoint and args.init_pretraining_params:
            logger.info(
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

    test_exe = exe
    if args.do_val or args.do_test:
        if args.use_multi_gpu_test:
            test_exe = fluid.ParallelExecutor(
                use_cuda=args.use_cuda,
                main_program=test_prog,
                share_vars_from=train_exe)

    if args.do_train:
        train_pyreader.start()
        steps = 0
        if warmup_steps > 0:
            graph_vars["learning_rate"] = scheduled_lr

        time_begin = time.time()

        skip_steps = args.skip_steps
        while True:
            try:
                steps += 1

                if steps % skip_steps == 0:
                    outputs = evaluate(args=args, exe=train_exe, program=train_program,
                                       pyreader=train_pyreader, graph_vars=graph_vars,
                                       eval_phase="train", vocab_size=vocab_size)

                    if args.verbose:
                        verbose = "train pyreader queue size: %d, " % train_pyreader.queue.size()
                        verbose += "learning rate: %f" % (
                            outputs["learning_rate"]
                            if warmup_steps > 0 else args.learning_rate)
                        logger.info(verbose)

                    current_example, current_epoch = reader.get_train_progress()
                    time_end = time.time()
                    used_time = time_end - time_begin
                    logger.info("epoch: %d, progress: %d/%d, step: %d, loss: %f, "
                                "ppl: %f, acc: %f, learning rate: %.8f, speed: %f steps/s"
                                % (current_epoch, current_example, num_train_examples,
                                   steps, outputs["loss"], outputs["ppl"], outputs["acc"],
                                   outputs["learning_rate"] if warmup_steps > 0 else args.learning_rate,
                                   args.skip_steps / used_time))
                    time_begin = time.time()
                else:
                    train_exe.run(fetch_list=[])

                if nccl2_trainer_id == 0:
                    if steps % args.save_steps == 0:
                        save_path = os.path.join(args.checkpoints,
                                                 "step_" + str(steps))
                        fluid.io.save_persistables(exe, save_path, train_program)

                    if steps % args.validation_steps == 0:
                        # evaluate dev set
                        if args.do_val:
                            test_pyreader.decorate_tensor_provider(
                                reader.data_generator(
                                    args.dev_set,
                                    batch_size=args.batch_size,
                                    epoch=1,
                                    dev_count=1,
                                    shuffle=False,
                                    phase='dev',
                                    do_dec=args.do_dec,
                                    place=place))
                            evaluate(args=args, exe=test_exe, program=test_prog, pyreader=test_pyreader,
                                     graph_vars=test_graph_vars, eval_phase="dev",
                                     vocab_size=vocab_size, do_dec=args.do_dec,
                                     vocab_path=args.vocab_path, features=reader.get_features("dev"),
                                     decode_path=args.decode_path + "/valid_" + str(steps) + "_preds")
                        # evaluate test set
                        if args.do_test:
                            test_pyreader.decorate_tensor_provider(
                                reader.data_generator(
                                    args.test_set,
                                    batch_size=args.batch_size,
                                    epoch=1,
                                    dev_count=1,
                                    shuffle=False,
                                    phase='test',
                                    do_dec=args.do_dec,
                                    place=place))
                            evaluate(args=args, exe=test_exe, program=test_prog, pyreader=test_pyreader,
                                     graph_vars=test_graph_vars, eval_phase="test",
                                     vocab_size=vocab_size, do_dec=args.do_dec,
                                     vocab_path=args.vocab_path, features=reader.get_features("test"),
                                     decode_path=args.decode_path + "/test_" + str(steps) + "_preds")

            except fluid.core.EOFException:
                save_path = os.path.join(args.checkpoints, "step_" + str(steps))
                fluid.io.save_persistables(exe, save_path, train_program)
                train_pyreader.reset()
                break

    if nccl2_trainer_id == 0:
        # final eval on dev set
        if args.do_val:
            test_pyreader.decorate_tensor_provider(
                reader.data_generator(
                    args.dev_set,
                    batch_size=args.batch_size,
                    epoch=1,
                    dev_count=1,
                    shuffle=False,
                    phase='dev',
                    do_dec=args.do_dec,
                    place=place))
            logger.info("Final validation result:")
            evaluate(args=args, exe=test_exe, program=test_prog, pyreader=test_pyreader,
                     graph_vars=test_graph_vars, eval_phase="dev",
                     vocab_size=vocab_size, do_dec=args.do_dec,
                     vocab_path=args.vocab_path, features=reader.get_features("dev"),
                     decode_path=args.decode_path + "/valid_final_preds")

        # final eval on test set
        if args.do_test:
            test_pyreader.decorate_tensor_provider(
                reader.data_generator(
                    args.test_set,
                    batch_size=args.batch_size,
                    epoch=1,
                    dev_count=1,
                    shuffle=False,
                    phase='test',
                    do_dec=args.do_dec,
                    place=place))
            logger.info("Final test result:")
            evaluate(args=args, exe=test_exe, program=test_prog, pyreader=test_pyreader,
                     graph_vars=test_graph_vars, eval_phase="test",
                     vocab_size=vocab_size, do_dec=args.do_dec,
                     vocab_path=args.vocab_path, features=reader.get_features("test"),
                     decode_path=args.decode_path + "/test_final_preds")


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the beam-search decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def evaluate_bleu(refs, preds, bleu_n=4):
    """Compute Blue score"""
    eval_res = compute_bleu(refs, preds, max_order=bleu_n)
    return eval_res[0]


def report_rouge(gold_path, can_path):
    """Calculating Rouge"""
    logger.info("Calculating Rouge")
    candidates = codecs.open(can_path, encoding="utf-8")
    references = codecs.open(gold_path, encoding="utf-8")
    results_dict = test_rouge(candidates, references, 5)
    return results_dict


def evaluate(args, exe, program, pyreader, graph_vars, eval_phase, vocab_size,
             do_dec=False, vocab_path=None, features=None, decode_path=""):
    """Obtain model loss or decoding output"""

    if args.label_smooth_eps:
        # the best cross-entropy value with label smoothing
        loss_normalizer = -(
                (1. - args.label_smooth_eps) * np.log((1. - args.label_smooth_eps)) +
                args.label_smooth_eps * np.log(args.label_smooth_eps / (vocab_size - 1) + 1e-20))
    else:
        loss_normalizer = 0.0

    if do_dec and not hasattr(evaluate, 'spm_vocab'):
        """load vocabulary"""
        spm = sentencepiece.SentencePieceProcessor()
        spm.Load(vocab_path)
        symbols = {'BOS': spm.PieceToId('<S>'), 'EOS': spm.PieceToId('</S>'), 'PAD': spm.PieceToId('<PAD>'),
                   'EOT': spm.PieceToId('<T>'), 'EOP': spm.PieceToId('<P>'), 'EOQ': spm.PieceToId('<Q>'),
                   'UNK': spm.PieceToId('<UNK>')}
        logger.info(symbols)
        evaluate.spm_vocab = spm
        evaluate.symbols = symbols

    if eval_phase == "train":
        fetch_list = [
            graph_vars["loss"].name,
            graph_vars["sum_correct"].name,
            graph_vars["token_num"].name
        ]
        if "learning_rate" in graph_vars:
            fetch_list.append(graph_vars["learning_rate"].name)
        outputs = exe.run(fetch_list=fetch_list)

        sum_cost_val = outputs[0]
        sum_correct_val = outputs[1]
        token_num_val = outputs[2]
        # sum the cost from multi-devices
        total_avg_cost = np.mean(sum_cost_val)
        total_token_num = token_num_val.sum()
        total_correct = sum_correct_val.sum()
        total_acc = (total_correct / total_token_num) * 100

        ret = {
            "loss": total_avg_cost - loss_normalizer,
            "ppl": np.exp(total_avg_cost - loss_normalizer),
            "acc": total_acc
        }
        if "learning_rate" in graph_vars:
            ret["learning_rate"] = float(outputs[3][0])
        return ret

    if not do_dec:
        fetch_list = [
            graph_vars["loss"].name,
            graph_vars["sum_correct"].name,
            graph_vars["token_num"].name
        ]
    else:
        fetch_list = [
            graph_vars["finished_ids"].name,
            graph_vars["finished_scores"].name,
            graph_vars["data_ids"].name,
        ]

    if do_dec:
        return_numpy = False
        dec_out = {}
    else:
        steps = 0
        cost = 0.0
        acc = 0.0
        return_numpy = True

    time_begin = time.time()
    pyreader.start()

    while True:
        try:
            if args.use_multi_gpu_test:
                outputs = exe.run(fetch_list=fetch_list,
                                  return_numpy=return_numpy)
            else:
                outputs = exe.run(program=program, fetch_list=fetch_list,
                                  return_numpy=return_numpy)

            if not do_dec:
                sum_cost_val = outputs[0]
                sum_correct_val = outputs[1]
                token_num_val = outputs[2]
                # sum the cost from multi-devices
                total_avg_cost = np.mean(sum_cost_val)
                total_token_num = token_num_val.sum()
                total_correct = sum_correct_val.sum()
                total_acc = (total_correct / total_token_num) * 100

                cost += total_avg_cost - loss_normalizer
                acc += total_acc
                steps += 1
            else:
                seq_ids, seq_scores, data_ids = outputs
                seq_ids_list, seq_scores_list = [seq_ids], [
                    seq_scores] if isinstance(
                    seq_ids, paddle.fluid.core.LoDTensor) else (seq_ids, seq_scores)

                data_ids = np.array(data_ids).reshape(-1).tolist()
                data_idx = 0

                for seq_ids, seq_scores in zip(seq_ids_list, seq_scores_list):
                    # How to parse the results:
                    #   Suppose the lod of seq_ids is:
                    #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
                    #   then from lod[0]:
                    #     there are 2 source sentences, beam width is 3.
                    #   from lod[1]:
                    #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
                    #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
                    # hyps = [[] for i in range(len(seq_ids.lod()[0]) - 1)]
                    # scores = [[] for i in range(len(seq_scores.lod()[0]) - 1)]
                    for i in range(len(seq_ids.lod()[0]) - 1):  # for each source sentence
                        start = seq_ids.lod()[0][i]
                        end = seq_ids.lod()[0][i + 1]
                        for j in range(end - start):  # for each candidate
                            sub_start = seq_ids.lod()[1][start + j]
                            sub_end = seq_ids.lod()[1][start + j + 1]
                            token_ids = [int(idx) for idx in post_process_seq(
                                np.array(seq_ids)[sub_start:sub_end],
                                evaluate.symbols['BOS'], evaluate.symbols['EOS'])]
                            print(len(token_ids))
                            hyp_str = evaluate.spm_vocab.DecodeIds(token_ids).replace(' ##', '').replace('<S>', ''). \
                                replace('</S>', '').replace('<Q>', '<q>').replace('<P>', ' '). \
                                replace('<T>', '').replace('<PAD>', '').replace('⁇', '"')
                            hyp_str = re.sub('\\s+', ' ', hyp_str)
                            print(hyp_str)

                            score = np.array(seq_scores)[sub_end - 1]
                            print(score)
                            data_id = data_ids[data_idx]
                            data_idx += 1
                            dec_out[data_id] = (hyp_str, score)

                            break

        except fluid.core.EOFException:
            pyreader.reset()
            break

    time_end = time.time()
    if not do_dec:
        logger.info(
            "[%s evaluation] loss: %f, ppl: %f, acc: %f, elapsed time: %f s"
            % (eval_phase, cost / steps, np.exp(cost / steps), acc / steps, time_end - time_begin))
    else:
        # start predicting
        gold_path = decode_path + '.gold'
        can_path = decode_path + '.candidate'
        gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        can_out_file = codecs.open(can_path, 'w', 'utf-8')

        preds = []
        refs = []
        keys = features.keys()

        for i in keys:
            ref_str = evaluate.spm_vocab.DecodeIds(
                post_process_seq(features[i].tgt, evaluate.symbols['BOS'], evaluate.symbols['EOS'])). \
                replace(' ##', '').replace('<S>', '').replace('</S>', '').replace('<Q>', '<q>').replace('<P>', ' '). \
                replace('<T>', '').replace('<PAD>', '').replace('⁇', '"')
            ref_str = re.sub('\\s+', ' ', ref_str)

            refs.append(ref_str)
            preds.append(dec_out[i][0])

            # logger.info("scores[i] = %.4f" % dec_out[i][1])
            gold_out_file.write(refs[i] + '\n')
            can_out_file.write(preds[i] + '\n')

        gold_out_file.close()
        can_out_file.close()

        if args.evaluate_blue:
            bleu = evaluate_bleu(refs, preds)
            logger.info(
                "[%s evaluation] bleu-4: %f, elapsed time: %f s"
                % (eval_phase, bleu, time_end - time_begin))

        if args.report_rouge:
            rouges = report_rouge(gold_path, can_path)
            logger.info('Rouges \n%s' % rouge_results_to_str(rouges))
            logger.info('elapsed time: %f s' % (time_end - time_begin))


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    init_logger(args.log_file)
    check_gpu(args.use_cuda)
    main(args)
