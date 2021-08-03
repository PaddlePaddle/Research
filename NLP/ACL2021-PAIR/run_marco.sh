#!/bin/bash
# This script contains the entire process of PAIR. You can reproduce the results of the paper base on these processes.
# Each part depends on the result of the previous step, and starts from the root directory.

ERNIE_BASE_DE='../checkpoint/ernie_base_twin_init/params'
DATA_PATH='../corpus/marco'

### Pre-Training
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRAIN_SET='../data_train/marco_pretrain.tsv'
sh script/run_dual_encoder_train.sh $TRAIN_SET $ERNIE_BASE_DE 8 true true true
# trained model -> [marco_pretrained_encoder]

### Fine-Tuning
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRAIN_SET='../data_train/marco_finetune.tsv'
MODEL_PATH='marco_pretrained_encoder'
sh script/run_dual_encoder_train.sh $TRAIN_SET $MODEL_PATH 8 true true false
# trained model -> [marco_finetuned_encoder]

### Inference on test set
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEST_SET='../corpus/marco/dev.query.txt'
MODEL_PATH='../checkpoint/marco_finetuned_encoder'
TOP_K=1000
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K

### Evaluation
recall_topk_dev='model/output/res.top1000'
python metric/msmarco_eval.py corpus/marco/qrels.dev.tsv $recall_topk_dev
