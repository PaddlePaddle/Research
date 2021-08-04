#!/bin/bash
# This script contains the entire process of PAIR. You can reproduce the results of the paper base on these processes.
# Each part depends on the result of the previous step, and starts from the root directory.

ERNIE_BASE_DE='../checkpoint/ernie_base_twin_init/params'
DATA_PATH='../corpus/nq'

### Pre-Training
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRAIN_SET='../data_train/nq_pretrain.tsv'
sh script/run_dual_encoder_train.sh $TRAIN_SET $ERNIE_BASE_DE 8 false false true
# trained model -> [nq_pretrained_encoder]

### Fine-Tuning
cd model
export CUDA_VISIBLE_DEVICES=0
TRAIN_SET='../data_train/nq_finetune.tsv'
MODEL_PATH='nq_pretrained_encoder'
sh script/run_dual_encoder_train.sh $TRAIN_SET $MODEL_PATH 1 false false false
# trained model -> [nq_finetuned_encoder]

### Inference on test set
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEST_SET='../corpus/nq/test.query.txt'
MODEL_PATH='../checkpoint/nq_finetuned_encoder'
TOP_K=100
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K

### Evaluation
recall_topk_dev='model/output/res.top100'
python metric/nq_eval.py $recall_topk_dev
