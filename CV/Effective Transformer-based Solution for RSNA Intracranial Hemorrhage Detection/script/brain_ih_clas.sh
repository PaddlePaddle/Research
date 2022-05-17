# Training
export FLAGS_fast_eager_deletion_mode=1
nohup python3 -u -m paddle.distributed.launch train.py \
    --config config/brain_intracranial_hemorrhage_clas_sequential.yml \
    --num_workers 4 \
    --save_interval 1000 \
    --resume_model ./model_zoo/0.04368_rank1/model.pdparams \
    --multigpu_infer \
    --overwrite_save_dir > log.log 2>&1 &

# Prediction
## Download weights

mkdir ./model_zoo/ && cd ./model_zoo/

wget "https://bj.bcebos.com/v1/ai-studio-online/01bf9a847d2b4e86a285b2e729dafb6c280aad1224eb468bba27f029cfbac336?responseContentDisposition=attachment%3B%20filename%3Dbrain_ihd.tar.gz&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-04-07T07%3A18%3A25Z%2F-1%2F%2F2eedf53628e35452d059aeed44c268d45a495105fab519dee5f446e821a311ee" -O brain_ihd.tar.gz # ~ 1.07GB

tar zxvf brain_ihd.tar.gz # ~ 1Gb
rm brain_ihd.tar.gz

## Pred 0.04368 rank1
python3 -u -m paddle.distributed.launch --log_dir infer_log infer.py \
        --config config/brain_intracranial_hemorrhage_clas_sequential.yml \
        --num_workers 4 \
        --infer_splition test \
        --resume_model ./model_zoo/0.04368_rank1/model.pdparams \
        --multigpu_infer

## Pred 0.04407 rank1
python3 -u -m paddle.distributed.launch --log_dir infer_log infer.py \
        --config config/brain_intracranial_hemorrhage_clas_sequential.yml \
        --num_workers 4 \
        --infer_splition test \
        --resume_model ./model_zoo/0.04407_rank2/model.pdparams \
        --multigpu_infer

## Pred 0.05117 rank 33
python3 -u -m paddle.distributed.launch --log_dir infer_log infer.py \
        --config config/brain_intracranial_hemorrhage_clas_sequential.yml \
        --num_workers 4 \
        --infer_splition test \
        --resume_model ./model_zoo/0.05117_rank33/model.pdparams \
        --multigpu_infer