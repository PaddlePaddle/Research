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
RSNA Intracranial Hemorrhage Detection 脑CT出血分类 数据预处理
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from easymia import datasets
from easymia.datasets.classification.rsna_ihd import BrainIntracranialHemorrhage as dataset

if __name__ == '__main__':
    root_dir = "/ssd2/data/open_source/brain/RSNA_intracranial_hemorrhage/"
    dest_dir = "/ssd2/easymia_processed_data/brain_ihd_clas/"

    train_dicom_dir = os.path.join(root_dir, "stage_2_train")
    test_dicom_dir = os.path.join(root_dir, "stage_2_test")

    train_label_path = os.path.join(root_dir, "stage_2_train.csv")

    train_meta_savepath = os.path.join(dest_dir, "train_meta.csv")
    val_meta_savepath = os.path.join(dest_dir, "val_meta.csv")
    test_meta_savepath = os.path.join(dest_dir, "test_meta.csv")

    # save as npy
    image_savepath = os.path.join(dest_dir, "imgdata")

    val_ratio = 0.1
    random_seed = 42

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    if not os.path.exists(image_savepath):
        os.mkdir(image_savepath)

    train_dicoms = os.listdir(train_dicom_dir)
    test_dicoms = os.listdir(test_dicom_dir)

    # Label
    train_label = pd.read_csv(train_label_path)
    train_label['slice_id'] = train_label['ID'].apply(lambda x: "ID_" + x.split("_")[1])
    train_label['disease'] = train_label['ID'].apply(lambda x: x.split("_")[2])
    ## drop duplicated rows
    train_label = train_label.loc[~train_label.duplicated(['slice_id', "disease", "Label"])]
    ## fuse diseases label into one row
    train_label = train_label.groupby("slice_id")['Label'].apply(lambda x: [int(i) for i in x]).reset_index()

    # Metainfo
    train_meta_info = Parallel(n_jobs=24, verbose=1)(
        (delayed(dataset.get_dicom_metainfo)(os.path.join(train_dicom_dir, p)) for p in train_dicoms))
    test_meta_info = Parallel(n_jobs=24, verbose=1)(
        (delayed(dataset.get_dicom_metainfo)(os.path.join(test_dicom_dir, p)) for p in test_dicoms))

    ## slice_id, series_id, study_id, patient_id, window_center, window_width, intercept, slope, position
    train_meta_info = pd.DataFrame(train_meta_info)
    test_meta_info = pd.DataFrame(test_meta_info)
    ## sort slices by series_id + slice_position
    train_meta_info = train_meta_info.sort_values(['series_id', 'ImagePos1', "ImagePos2", "ImagePos3"])
    test_meta_info = test_meta_info.sort_values(['series_id', 'ImagePos1', "ImagePos2", "ImagePos3"])

    train_meta_info = train_meta_info.merge(train_label[['slice_id', 'Label']], how='left', on='slice_id')

    train_meta_info['slice_idx'] = train_meta_info.groupby(['series_id']).cumcount()
    test_meta_info['slice_idx'] = test_meta_info.groupby(['series_id']).cumcount()
    ## split train val
    train_series, val_series = train_test_split(train_meta_info.series_id.unique(), 
                                                    test_size=val_ratio, 
                                                    random_state=random_seed)

    val_meta_info = train_meta_info.loc[train_meta_info.series_id.isin(val_series)]
    train_meta_info = train_meta_info.loc[train_meta_info.series_id.isin(train_series)]
    # save csv
    train_meta_info.to_csv(train_meta_savepath, index=False)
    val_meta_info.to_csv(val_meta_savepath, index=False)
    test_meta_info.to_csv(test_meta_savepath, index=False)

    # Image Dicom -> .npy
    _ = Parallel(n_jobs=24, verbose=0)(
        (delayed(dataset.preprocess)(series_id, rows.slice_id.tolist(), train_dicom_dir, savepath=image_savepath) 
        for series_id, rows in train_meta_info.groupby("series_id")))

    _ = Parallel(n_jobs=24, verbose=0)(
        (delayed(dataset.preprocess)(series_id, rows.slice_id.tolist(), train_dicom_dir, savepath=image_savepath) 
        for series_id, rows in val_meta_info.groupby("series_id")))

    _ = Parallel(n_jobs=24, verbose=0)(
        (delayed(dataset.preprocess)(series_id, rows.slice_id.tolist(), train_dicom_dir, savepath=image_savepath)
        for series_id, rows in test_meta_info.groupby("series_id")))