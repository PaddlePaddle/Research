#!/bin/bash
# This script contains the entire process of RocketQA. You can reproduce the results of the paper base on these processes.
# Each part depends on the result of the previous step, and starts from the root directory. Since we have given some processed training data and trained models, you can start from any step.

ERNIE_BASE_DE='../checkpoint/ernie_base_twin_init/params'
ERNIE_LARGE_CE='../checkpoint/ernie_large_en/params'
DATA_PATH='../corpus/marco'

### STEP1-Train
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRAIN_SET='../data_train/marco_de0_denoise.tsv'
sh script/run_dual_encoder_train.sh $TRAIN_SET $ERNIE_BASE_DE 40 8
# trained model -> [marco_dual_encoder_v0]

### STEP1-Inference(get recall of train set)
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEST_SET='../corpus/marco/train.query.txt'
MODEL_PATH='../checkpoint/marco_dual_encoder_v0'
TOP_K=1000
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K

### STEP2-Data process
recall_topk_trainset='model/output/res.top1000'
output_file_ce='data_train/marco_ce0.tsv'
python data_process/construct_marco_train_ce.py $recall_topk_trainset $output_file_ce

### STEP2-Train
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3
TRAIN_SET=../$output_file_ce
MODEL_PATH=$ERNIE_LARGE_CE
sh script/run_cross_encoder_train.sh $TRAIN_SET $MODEL_PATH 2 4
# trained model -> [marco_cross_encoder_large]

### STEP2-Inference(get score of cross-encoder on top-k)
cd model
export CUDA_VISIBLE_DEVICES=0
TEST_SET='recall_top50_trainset.tsv' # use $recall_topk_trainset to get top50 textual query-paras.
MODEL_PATH='../checkpoint/marco_cross_encoder_large'
sh script/run_cross_encoder_test.sh $TEST_SET $MODEL_PATH

### STEP3-Data process
ce_score_topk_file='model/output/recall_top50_trainset.tsv.score'
output_file_de1='data_train/marco_de1_denoise.tsv'
python data_process/construct_marco_train_de.py $recall_topk_trainset $ce_score_topk_file $output_file_de1

### STEP3-Train
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRAIN_SET=../$output_file_de1
MODEL_PATH=$ERNIE_BASE_DE
sh script/run_dual_encoder_train.sh $TRAIN_SET $MODEL_PATH 10 8
# *To get better performence, you can use trained model of STEP1 as initial parameters, and swtich off 'cross-encoder' (set --use_cross_batch False in script):
# MODEL_PATH='../checkpoint/marco_dual_encoder_v0'
# sh script/run_dual_encoder_train.sh $TRAIN_SET $MODEL_PATH 10 8
# trained model -> [marco_dual_encoder_v1]

### STEP3-Inference(get recall of augment data set)
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEST_SET='../corpus/augment/orcas_yahoo_nq.query.txt'
MODEL_PATH='../checkpoint/marco_dual_encoder_v1'
TOP_K=50
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K

### STEP3-Inference(get score of cross-encoder on top-k)
recall_topk_augset='model/output/res.top50'
cd model
export CUDA_VISIBLE_DEVICES=0
TEST_SET='recall_top50_augset.tsv' # use $recall_topk_augset to get top50 textual query-paras.
MODEL_PATH='../checkpoint/marco_cross_encoder_large'
sh script/run_cross_encoder_test.sh $TEST_SET $MODEL_PATH

### STEP4-Data process
ce_score_topk_augset='model/output/recall_top50_augset.tsv.score'
output_file_aug='data_train/marco_unlabel_de2_denoise.tsv'
python data_process/construct_unlabel_train_de.py $recall_topk_augset $ce_score_topk_augset $output_file_aug marco
output_file_de2='data_train/marco_merge_de2_denoise.tsv'
cat $output_file_de1 $output_file_aug > $output_file_de2

### STEP4-Train
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TRAIN_SET=../$output_file_de2
MODEL_PATH=$ERNIE_BASE_DE
sh script/run_dual_encoder_train.sh $TRAIN_SET $MODEL_PATH 10 8
# trained model -> [marco_dual_encoder_v2]

### STEP4-Inference on test set
cd model
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TEST_SET='../corpus/marco/dev.query.txt'
MODEL_PATH='../checkpoint/marco_dual_encoder_v2'
TOP_K=1000
sh script/run_retrieval.sh $TEST_SET $MODEL_PATH $DATA_PATH $TOP_K

### Evaluation
recall_topk_dev='model/output/res.top1000'
python metric/msmarco_eval.py corpus/marco/qrels.dev.tsv $recall_topk_dev
