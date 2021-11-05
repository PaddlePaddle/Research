#!/usr/bin/env bash
set -x

PRED=$1
PREFIX=$2

python pyrouge_set_rouge_path.py `pwd`/file2rouge/
python cnndm/eval.py --pred ${PRED} \
  --gold ${PREFIX} --trunc_len 250 --perl
