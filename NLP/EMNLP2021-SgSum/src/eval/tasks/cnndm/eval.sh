#!/usr/bin/env bash
PRED=`pwd`$1
REF=`pwd`$2

cd `dirname $0`
bash cnndm_eval.sh $PRED $REF | grep ROUGE-F | awk -F ": " '{print $2}' | awk -F "/" '{print "{\"rouge-1\": "$1", \"rouge-2\": "$2", \"rouge-l\": "$3"}"}'