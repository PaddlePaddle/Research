#!/bin/bash

TASK=jf17k
#==========dataset configurations
VOCAB_SIZE=29148
NUM_RELATIONS=501
MAX_SEQ_LEN=11
MAX_ARITY=6

#==========tunable hyperparameters
ENTITY_SOFT_LABEL=0.1
RELATION_SOFT_LABEL=1.0
DROPOUT_PROB=0.2
EPOCH=160

#=========paths for training & evaluation
OUTPUT="./${TASK}_out"
TRAIN_FILE="./data/${TASK}/train.json"
PREDICT_FILE="./data/${TASK}/test.json"
GROUND_TRUTH_PATH="./data/${TASK}/all.json"
VOCAB_PATH="./data/${TASK}/vocab.txt"
CHECKPOINTS=$OUTPUT
LOG_FILE="$OUTPUT/train.log"


