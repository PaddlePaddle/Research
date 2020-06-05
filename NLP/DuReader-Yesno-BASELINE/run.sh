set -ex

if [ -z "$CUDA_VISIBLE_DEVICES" ];then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ ! -d output ]; then
    mkdir output
fi

export FLAGS_eager_delete_tensor_gb=1
export FLAGS_sync_nccl_allreduce=1

python -u src/run_classifier.py $@
