#!/bin/bash
# Download dataset and model parameters
set -e

echo "Download DuReader-checklist dataset"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/lic2021/dureader_checklist.dataset.tar.gz
tar -zxvf dureader_checklist.dataset.tar.gz
rm dureader_checklist.dataset.tar.gz

echo "Download fine-tuned parameters"
wget --no-check-certificate https://dataset-bj.cdn.bcebos.com/lic2021/dureader_checklist.finetuned_model.tar.gz
tar -zxvf dureader_checklist.finetuned_model.tar.gz
rm dureader_checklist.finetuned_model.tar.gz
