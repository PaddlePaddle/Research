export CUDA_VISIBLE_DEVICES=$1
nohup python -u run.py train experiments/spider-label-smooth-bert-large-run.jsonnet >aug.log &2>1 &
