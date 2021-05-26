#!/bin/bash

TASK=wikipeople
#==========dataset configurations
VOCAB_SIZE=47960
NUM_RELATIONS=193
MAX_SEQ_LEN=17
MAX_ARITY=9

#==========tunable hyperparameters
ENTITY_SOFT_LABEL=0.8
RELATION_SOFT_LABEL=0.8
DROPOUT_PROB=0.1
EPOCH=200

#=========paths for training & evaluation
OUTPUT="./${TASK}_out"
TRAIN_FILE="./data/${TASK}/train+valid.json"
PREDICT_FILE="./data/${TASK}/test.json"
GROUND_TRUTH_PATH="./data/${TASK}/all.json"
VOCAB_PATH="./data/${TASK}/vocab.txt"
CHECKPOINTS=$OUTPUT
LOG_FILE="$OUTPUT/train.log"

