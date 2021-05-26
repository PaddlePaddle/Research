#!/bin/bash

TASK=jf17k-3
#==========dataset configurations
VOCAB_SIZE=11751
NUM_RELATIONS=208
MAX_SEQ_LEN=5
MAX_ARITY=3

#==========tunable hyperparameters
ENTITY_SOFT_LABEL=0.2
RELATION_SOFT_LABEL=0.8
DROPOUT_PROB=0.2
EPOCH=180

#=========paths for training & evaluation
OUTPUT="./${TASK}_out"
TRAIN_FILE="./data/${TASK}/train+valid.json"
PREDICT_FILE="./data/${TASK}/test.json"
GROUND_TRUTH_PATH="./data/${TASK}/all.json"
VOCAB_PATH="./data/${TASK}/vocab.txt"
CHECKPOINTS=$OUTPUT
LOG_FILE="$OUTPUT/train.log"

