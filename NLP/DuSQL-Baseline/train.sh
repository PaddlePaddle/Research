#!/bin/bash

WORKROOT=$(cd $(dirname $0); pwd)
cd $WORKROOT

if [ $# -lt 1 ]; then
    echo "usage:"
    echo "    $0 <lstm|ernie> [train options]"
    exit 1
fi

encoder=$1
shift
echo "running model with $encoder encoder"

if [ $encoder == 'lstm' ]; then
    ## run lstm encoder version
    bash run.sh ./script/text2sql_train.py --config ./conf/train_text2sql_basic.json $@
elif [ $encoder == 'ernie' ]; then
    ## run ernie encoder version
    bash run.sh ./script/text2sql_train.py --config ./conf/train_text2sql_ernie.json $@
fi

