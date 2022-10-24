#!/bin/bash
set -eux

DATASET="SMD"
MODEL="T5-Large"

python3 infer.py \
    --infer_file "./data/${DATASET}/test.json" \
    --model_path "./models/Q-TOD_${MODEL}_${DATASET}/" \
    --batch_size 4 \
    --beam_size 4 \
    --save_file "./output/${MODEL}_${DATASET}/inference_output.json"

python3 evaluate.py \
    --dataset ${DATASET} \
    --pred_file "./output/${MODEL}_${DATASET}/inference_output.json" \
    --entity_file "./data/${DATASET}/entities.json" \
    --save_file "./output/${MODEL}_${DATASET}/results.json"

exit_code=$?
exit $exit_code
