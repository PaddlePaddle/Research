#!/bin/bash

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1

export DATA_PATH=./data/
export MODEL_PATH=./pretrained_lm/ernie2_base_en

ckpt_path=./checkpoints/step_xxxx
predict_thresh=xxxx
with_ent_structure=true
batch_size=4 # WARNING: this should be consistent with your checkpoint training batchsize.

CUDA_VISIBLE_DEVICES=0 python ./run_ssan.py                              \
       --use_cuda true                                                   \
       --use_fast_executor ${e_executor:-"true"}                         \
       --tokenizer ${TOKENIZER:-"FullTokenizer"}                         \
       --use_fp16 ${USE_FP16:-"false"}                                   \
       --do_train false                                                  \
       --do_val false                                                    \
       --do_test true                                                    \
       --predict_thresh ${predict_thresh}                                \
       --with_ent_structure ${with_ent_structure}                        \
       --model_path ${MODEL_PATH}                                        \
       --init_checkpoint ${ckpt_path}                                    \
       --data_path ${DATA_PATH}                                          \
       --batch_size ${batch_size}                                        \
       --max_seq_len 512                                                 \
       --max_ent_cnt 42                                                  \
       --num_labels 97                                                   \
       --weight_decay  0.0
