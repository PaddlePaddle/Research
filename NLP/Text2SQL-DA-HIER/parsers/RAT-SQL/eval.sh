export CUDA_VISIBLE_DEVICES=$1
nohup python run.py eval $2 > $3.log &2>1 &