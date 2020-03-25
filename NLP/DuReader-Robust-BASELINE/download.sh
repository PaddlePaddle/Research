#!/bin/bash
# Download dataset and model parameters
set -e

echo "Download ERNIE 1.0"
mkdir pretrained_model
cd pretrained_model
wget --no-check-certificate https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz
tar -zxvf ERNIE_1.0_max-len-512.tar.gz
rm ERNIE_1.0_max-len-512.tar.gz
cd ..

echo "Download DuReader-robust dataset"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader_robust/data/dureader_robust-data.tar.gz 
tar -zxvf dureader_robust-data.tar.gz 
mv dureader_robust-data data
rm dureader_robust-data.tar.gz

echo "Download fine-tuned parameters"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader_robust/baseline/dureader_robust-baseline-finetuned.tar.gz
tar -zxvf dureader_robust-baseline-finetuned.tar.gz
rm dureader_robust-baseline-finetuned.tar.gz
