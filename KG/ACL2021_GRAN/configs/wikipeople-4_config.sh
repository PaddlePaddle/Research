#!/bin/bash

TASK=wikipeople-4
#==========dataset configurations
VOCAB_SIZE=9625
NUM_RELATIONS=95
MAX_SEQ_LEN=7
MAX_ARITY=4

#==========tunable hyperparameters
ENTITY_SOFT_LABEL=0.2
RELATION_SOFT_LABEL=0.6
DROPOUT_PROB=0.3
EPOCH=100

#=========paths for training & evaluation
OUTPUT="./${TASK}_out"
TRAIN_FILE="./data/${TASK}/train+valid.json"
PREDICT_FILE="./data/${TASK}/test.json"
GROUND_TRUTH_PATH="./data/${TASK}/all.json"
VOCAB_PATH="./data/${TASK}/vocab.txt"
CHECKPOINTS=$OUTPUT
LOG_FILE="$OUTPUT/train.log"


