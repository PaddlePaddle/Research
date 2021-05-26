#!/bin/bash

TASK=wikipeople-
#==========dataset configurations
VOCAB_SIZE=35005
NUM_RELATIONS=178
MAX_SEQ_LEN=13
MAX_ARITY=7

#==========tunable hyperparameters
ENTITY_SOFT_LABEL=0.8
RELATION_SOFT_LABEL=0.9
DROPOUT_PROB=0.1
EPOCH=160

#=========paths for training & evaluation
OUTPUT="./${TASK}_out"
TRAIN_FILE="./data/${TASK}/train+valid.json"
PREDICT_FILE="./data/${TASK}/test.json"
GROUND_TRUTH_PATH="./data/${TASK}/all.json"
VOCAB_PATH="./data/${TASK}/vocab.txt"
CHECKPOINTS=$OUTPUT
LOG_FILE="$OUTPUT/train.log"

