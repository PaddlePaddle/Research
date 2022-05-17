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
Brain Intracranial Hemorrhage  dataset
https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview
"""

import os
import numbers
from collections.abc import Sequence, Mapping

import cv2
import paddle
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk

from easymia.libs import manager
from easymia.datasets.classification.clas_dataset import ClasDataset
from easymia.transforms import functional as F

@manager.DATASETS.add_component
class BrainIntracranialHemorrhage(ClasDataset):
    """
    BrainIntracranialHemorrhage : a open source medical image dataset for Brain Intracranial Hemorrhage

    数据划分-------训练集:677525 验证集75281------
    train dataset length: 677525
    eval dataset length: 75282
    test dataset length: 121232
    """
    def __init__(self, split='train', transforms=None, dataset_root=None, sequential=False, tag=None):
        '''
        dataset_root = "/ssd2/easymia_processed_data/brain_ihd_clas/"
        '''
        super().__init__(split, transforms, dataset_root)
        self.sequential = sequential
        self.max_slice_len = 60
        if tag is None: tag = split
        self.data_list = self.get_data_list("{}_meta.csv".format(tag))

    def get_data_list(self, csv_path):
        """
        Function:
                generate the data list 
                data_list:[[ID_12cadc6af, [0, 0, 0, 0, 0], slice_idx], ......]
        """
        data_df = pd.read_csv(os.path.join(self.dataset_root, csv_path))

        if self.split in ['train', 'val']:
            data_df['Label'] = data_df['Label'].apply(lambda x: eval(x))
        else:
            data_df['Label'] = None

        if self.sequential:
            data_df = data_df.groupby("series_id")[["Label"]].apply(
                lambda rows: {k: row['Label'] for k, row in rows.iterrows()}).reset_index()
            data_df.columns = ['series_id', 'info']

            data_df['slice_idx'] = data_df['info'].apply(lambda x: list(x.keys()))
            data_df['Label'] = data_df['info'].apply(lambda x: np.array(list(x.values())))

        record_num = len(data_df)

        data_df['filepath'] = data_df['series_id'].apply(
            lambda x: os.path.join(self.dataset_root, "imgdata", x + ".npz"))
        data_df['exists'] = data_df['filepath'].apply(lambda x: os.path.exists(x))
        data_list = data_df.loc[data_df['exists'], ['filepath', 'Label', 'slice_idx']].values.tolist()

        final_record_num = len(data_list)
        print("Read {} records, {} records are valid.".format(record_num, final_record_num))
        return data_list

    def collate_fn(self):
        """
        default paddle.fluid.dataloader.collate.default_collate_fn
        """
        def func(batch):
            """
            Custom collate fn
            """
            sample = batch[0]
            if isinstance(sample, np.ndarray):
                if len(batch) == 1: return batch[0]
                else:
                    raise RuntimeError(
                        "You must set batch size = 1 when enable sequential mode")
            elif isinstance(sample, Mapping):
                return {
                    key: func([d[key] for d in batch])
                    for key in sample
                }
            elif isinstance(sample, numbers.Number):
                batch = np.array(batch)
                return batch
            elif isinstance(sample, Sequence):
                sample_fields_num = len(sample)
                if not all(len(sample) == sample_fields_num for sample in iter(batch)):
                    raise RuntimeError(
                        "fileds number not same among samples in a batch")
                return [func(fields) for fields in zip(*batch)]
    
        if not self.sequential:
            return paddle.fluid.dataloader.collate.default_collate_fn
        else:
            return func

    def __getitem__(self, idx):
        """
        getitem

        return: Single Image shape = [C, H, W], uint8
        """
        img_path, label, slice_idx = self.data_list[idx]
        img = np.load(img_path)['arr_0'] # CUBE, shape Slices, 512, 512, 3

        if not self.sequential:
            img = img[slice_idx]

        if self.transforms:
            if self.sequential:
                img = list(map(self.remove_black_edge, img))
                img = np.array(list(map(self.transforms, img)))
                img = img.transpose(0, 3, 1, 2)
            else:
                img = self.transforms(img)
                img = img.transpose(2, 0, 1)

        if self.split in ['val', 'test']:
            pad_length = self.max_slice_len - img.shape[0]
            img = np.pad(img, [[0, pad_length]] + [[0, 0]] * (img.ndim - 1), constant_values=0)
            idx = np.array(slice_idx)
            idx = np.pad(idx, [0, pad_length], constant_values=-1)

            if self.split == "val":
                label = np.pad(label, [[0, pad_length]] + [[0, 0]] * (label.ndim - 1), constant_values=255)

        label = np.array(label, dtype='float32')

        if self.split == 'test':
            return img, idx
        # train & eval
        elif self.split == "val":
            return {"data": img, "label": label, "index": idx}
        else:
            # cube_label = label.max(0, keepdims=True).astype("float32")
            return {"data": img, "label": label, "index": idx}

    def remove_black_edge(self, img):
        """
        去除脑CT切片图像中的黑边
        img: H, W, C
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close = cv2.morphologyEx(img[..., 0], cv2.MORPH_OPEN, kernel, iterations=3)
        if close.sum() > 0:
            (hmin, hmax), (wmin, wmax) = map(lambda x: (x.min(), x.max()), np.where(close > 0))
            return img[hmin:hmax, wmin:wmax]
        else:
            return np.zeros_like(img)

    def __len__(self):
        return len(self.data_list)

    @classmethod
    def get_dicom_metainfo(cls, img_path):
        """
        预处理utils
        """
        img_dicom = pydicom.read_file(img_path)
        metadata = {
            "slice_id": img_dicom.SOPInstanceUID,
            "series_id": img_dicom.SeriesInstanceUID,
            "study_id": img_dicom.StudyInstanceUID,
            "patient_id": img_dicom.PatientID,
            "window_center": img_dicom.WindowCenter,
            "window_width": img_dicom.WindowWidth,
            "intercept": img_dicom.RescaleIntercept,
            "slope": img_dicom.RescaleSlope,
        }

        for i in range(1, 4):
            metadata['ImagePos{}'.format(i)] = img_dicom.ImagePositionPatient[i - 1]
        return metadata

    @classmethod
    def dicom2img(cls, dcm_image, window_center, window_width):
        """
        将一张脑CT切片变换为一幅图像
        """
        assert (isinstance(window_center, int) and isinstance(window_width, int)) or \
                (len(window_center) == len(window_width))
        
        if isinstance(window_center, int):
            window_center = [window_center]
            window_width = [window_width]
            channel = 1
        else:
            channel = len(window_center)

        img = np.zeros(list(dcm_image.shape) + [channel], dtype="uint8")
        for i, (c, w) in enumerate(zip(window_center, window_width)):
            HU_min, HU_max = c - w // 2, c + w // 2
            img[..., i] = F.hu2uint8(dcm_image, HU_min=HU_min, HU_max=HU_max)
        return img

    @classmethod
    def preprocess(cls, series_id, slice_ids, dicom_dir, return_img=False, savepath=None):
        """
        image preprocess
        series_id: str, "ID_xxxxxxx"
        slice_ids: [str], ["ID_XXX", "ID_XXX", ...]
        """
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        dcm_list = [os.path.join(dicom_dir, i + ".dcm") for i in slice_ids]
        dcm_image, origin, spacing = F.load_dcm(dcm_list)
        # brain, subdural, bone
        img = cls.dicom2img(dcm_image, window_center=[40, 80, 40], window_width=[80, 200, 380])

        # remove black boundary
        box = np.where(img > 0)
        y_min, y_max, x_min, x_max = box[1].min(), box[1].max(), box[2].min(), box[2].max()
        img = img[:, y_min:y_max, x_min:x_max]

        if return_img: 
            return img
        else:
            np.savez_compressed(os.path.join(savepath, series_id), img)