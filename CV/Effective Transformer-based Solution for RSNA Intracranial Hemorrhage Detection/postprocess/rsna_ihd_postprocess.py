# -*-coding utf-8 -*-
##########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
##########################################################################
"""
RSNA Intracranial Hemorrhage Detection 脑CT出血分类 提交结果后处理
"""

import os
import pandas as pd

def make_submission(test_meta_path, pred_csv_path, submission_csv_path):
    """
    将模型预测记录csv变换为kaggle提交形式
    """
    data_df = pd.read_csv(test_meta_path)
    data_df['idx'] = data_df.index.copy()

    pred_file = pd.read_csv(pred_csv_path, names=['idx', 'probs'])
    pred_file = pred_file.loc[pred_file.idx != -1]
    pred_file[diseases] = pred_file['probs'].str.split(" ", expand=True).astype("float32")

    pred_file = pred_file.loc[~pred_file.duplicated()]

    collector = {}
    for k, row in data_df[['slice_id', 'idx', 'series_id']].merge(pred_file, how='left', on='idx').iterrows():
        for d in diseases:
            collector[row['slice_id'] + "_" + d] = row[d]

    result_df = pd.DataFrame.from_dict(collector, orient='index').reset_index()
    result_df.columns = ['ID', "Label"]
    result_df.to_csv(submission_csv_path, index=False)


def ensemble_submission(submission_paths, weights):
    """
    结果加权融合以获得更好的指标
    """
    results = pd.DataFrame(columns=['ID', "Label"])
    keys = []
    for i, csv in enumerate(submission_paths):
        data_frame = pd.read_csv(csv)
        data_frame.columns = ['ID', "Label{}".format(i)]
        results = results.merge(data_frame, how='outer', on='ID')
        keys.append("Label{}".format(i))

    results['Label'] = results[keys].apply(lambda row: 
                        (row.values * weights).sum() if (row.values > 0.998).sum() < 3 else 1, axis=1)

    results[['ID', "Label"]].to_csv('0.04345_rank1.csv', index=False)

if __name__ == '__main__':
    """
    # Pred 0.04368 rank1
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -u -m paddle.distributed.launch --log_dir infer_log infer.py \
            --config configs/brain_ihd_cls/brain_intracranial_hemorrhage_clas_sequential.yml \
            --num_workers 4 \
            --infer_splition test \
            --resume_model  ./model_zoo/brain_ihd/0.04368_rank1/model.pdparams \
            --multigpu_infer

    # Pred 0.04407 rank1
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -u -m paddle.distributed.launch --log_dir infer_log infer.py \
            --config configs/brain_ihd_cls/brain_intracranial_hemorrhage_clas_sequential.yml \
            --num_workers 4 \
            --infer_splition test \
            --resume_model  ./model_zoo/brain_ihd/0.04407_rank2/model.pdparams \
            --multigpu_infer

    # Pred 0.05117 rank 33
    CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -u -m paddle.distributed.launch --log_dir infer_log infer.py \
            --config configs/brain_ihd_cls/brain_intracranial_hemorrhage_clas_sequential.yml \
            --num_workers 4 \
            --infer_splition test \
            --resume_model  ./model_zoo/brain_ihd/0.05117_rank33/model.pdparams \
            --multigpu_infer
    """
    diseases = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

    sub_04368 = ["./2022-04-07-14:43:41_csvwriter.csv", '0.04368_rank1.csv']
    sub_04407 = ["./2022-04-07-14:12:23_csvwriter.csv", '0.04407_rank2.csv']
    sub_05117 = ["./2022-04-07-14:23:17_csvwriter.csv", '0.05117_rank33.csv']
    test_meta_path = "/ssd3/easymia_processed_data/brain_ihd_clas/test_meta.csv"

    ensemble_subpaths = [item[1] for item in [sub_04368, sub_04407, sub_05117]]

    for item in [sub_04368, sub_04407, sub_05117]:
        print("Processing {}...".format(item[0]))
        make_submission(test_meta_path, item[0], item[1])
    
    print("Ensembing...")
    ensemble_submission(ensemble_subpaths, [0.85, 0.05, 0.1])