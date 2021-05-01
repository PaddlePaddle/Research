#!/bin/bash
set -x

if [ $# != 4 ];then
    echo "USAGE: sh script/run_retrieval.sh \$QUERY_FILE \$MODEL_PATH \$DATA_PATH \$TOP_K"
    exit 1
fi

QUERY_FILE=$1
MODEL_PATH=$2
DATA_PATH=$3
TOP_K=$4

sh script/run_dual_encoder_inference.sh 0 q $MODEL_PATH $QUERY_FILE.format

for card in {0..7};do
    nohup sh script/run_dual_encoder_inference.sh ${card} ${card} $MODEL_PATH $DATA_PATH &
    pid[$card]=$!
    echo $card start: pid=$! >> output/test.log
done
wait

for part in {0..7};do
    nohup python src/index_search.py $part $TOP_K $QUERY_FILE >> output/test.log &
done
wait

para_part_cnt=`cat $DATA_PATH/para_8part/part-00 | wc -l`
python src/merge.py $para_part_cnt $TOP_K >> output/test.log
