#!/bin/bash
# Download dataset and model parameters
set -e

echo "Download ERNIE 1.0"
mkdir ernie
cd ernie
wget --no-check-certificate https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz
tar -zxvf ERNIE_1.0_max-len-512.tar.gz
rm ERNIE_1.0_max-len-512.tar.gz
cd ..

echo "Download DuReader-yesno dataset"
# TODO: add data donwload link
# wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/dureader_robust/data/dureader_robust-data.tar.gz 
# tar -zxvf dureader_robust-data.tar.gz 
# mv dureader_robust-data data
# rm dureader_robust-data.tar.gz

echo "Download fine-tuned parameters"
wget --no-check-certificate "http://bj.bcebos.com/v1/ai-studio-online/59b27b6ab663464a95683963a68ba16a1995c679b6e44c5aa8056f47b3365944?responseContentDisposition=attachment%3B%20filename%3Dbaseline.tar.gz&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-02-21T05%3A49%3A52Z%2F-1%2F%2Fe719555655a0314555aed5d033b16b3154b16b8b9b8b5ffec712edc94e9ca1d3"
tar -zxvf baseline.tar.gz
rm baseline.tar.gz
