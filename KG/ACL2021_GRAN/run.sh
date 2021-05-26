#! /bin/bash

#==========
set -e
set -x
set -u
set -o pipefail
#==========

#==========configs
CONFIG_FILE=$1
CUDA_ID=$2

#==========init env
source $CONFIG_FILE
export CUDA_VISIBLE_DEVICES=$CUDA_ID
export FLAGS_sync_nccl_allreduce=1
# todo: replace with your own path
export LD_LIBRARY_PATH=$(pwd)/env/lib/nccl2.3.7_cuda9.0/lib:/home/work/cudnn/cudnn_v7/cuda/lib64:/home/work/cuda-9.0/extras/CUPTI/lib64/:/home/work/cuda-9.0/lib64/:$LD_LIBRARY_PATH

#==========output dir
if [ -d $OUTPUT ]; then
    rm -rf $OUTPUT
fi
mkdir $OUTPUT

#==========begin training
echo ">>begin training now"
python3 -u ./src/run.py \
 --use_cuda true \
 --do_train true \
 --do_predict true \
 --train_file $TRAIN_FILE \
 --predict_file $PREDICT_FILE \
 --ground_truth_path $GROUND_TRUTH_PATH \
 --vocab_path $VOCAB_PATH \
 --vocab_size $VOCAB_SIZE \
 --num_relations $NUM_RELATIONS \
 --max_seq_len $MAX_SEQ_LEN \
 --max_arity $MAX_ARITY \
 --hidden_dropout_prob $DROPOUT_PROB \
 --attention_dropout_prob $DROPOUT_PROB \
 --entity_soft_label $ENTITY_SOFT_LABEL \
 --relation_soft_label $RELATION_SOFT_LABEL \
 --epoch $EPOCH \
 --checkpoints $CHECKPOINTS > $LOG_FILE 2>&1


