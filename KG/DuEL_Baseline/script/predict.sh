MODEL_PATH="./pretrained_model/ERNIE_1.0_max-len-512"
TASK_DATA_PATH="./data/generated"
set -eux
DIR=$(cd $(dirname $0); pwd)
export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=${DIR}/../../../tools/python/bin/python
python -u ./ernie/infer_type_ranker.py \
             --use_cuda true \
             --batch_size 32 \
             --vocab_path ${MODEL_PATH}/vocab.txt \
             --init_checkpoint ./checkpoints/step_9000 \
             --label_map ${TASK_DATA_PATH}/type_label_map.json \
             --save_inference_model_path "./checkpoints/inference_model" \
             --do_lower_case true \
             --max_seq_len 128 \
             --ernie_config_path ${MODEL_PATH}/ernie_config.json \
             --do_predict true \
             --predict_set $TASK_DATA_PATH/test.txt \
             --num_labels 2
