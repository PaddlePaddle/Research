#!/usr/bin/env python
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File:signboard_dataset.py
func:招牌图像数据迭代器
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/08/02
"""
import os

import numpy as np
import cv2
from PIL import Image

import paddle
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.vision import transforms

class SignboardDataset(Dataset):
    """img geo fuse dataset
    Args:
        image_root: 图像存放文件夹
        cls_label_path: 数据路径及对应的label
        transform_ops: 数据增强op
    """
    def __init__(self, image_root, cls_label_path, transform_ops=None):
        super(SignboardDataset, self).__init__()
        self.image_root = image_root
        self.cls_label_path = cls_label_path
        self.transform_ops = transform_ops
        self.images = []
        self.labels = []

        self._load_anno()

    def _load_anno(self):
        """load data and label"""
        assert os.path.exists(self.cls_label_path)
        assert os.path.exists(self.image_root)

        with open(self.cls_label_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split('\t')
                self.images.append(l[2])
                self.labels.append(int(l[1]))

    def __getitem__(self, idx):
        """数据迭代
        Args:
            idx: 样本idx
        Returns:
            ret: 返回迭代数据 type:dict
        """
        ret = {}
        img = Image.open(os.path.join(self.image_root, self.images[idx])).convert('RGB')
        label = self.labels[idx]
       
        if self.transform_ops is not None:
            img = self.transform_ops(img)
        ret['img'] = img
        ret['label'] = label
        ret['word'] = ""
        ret['img_name'] = self.images[idx]
    
        return ret

    def __len__(self):
        """len
        Retutrns:
            返回数据集长度
        """
        return len(self.images)

    @property
    def class_num(self):
        """class num
        Retutrns:
            返回数据类别数
        """
        return len(set(self.labels))


if __name__ == '__main__':
    image_root = '/data3/data/data_home/sb_match_data2.4'
    cls_label_path = '/home/yuwei/code/paddle/MultiModelFuse-paddle/data/signboard/taojin-test'
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],# 取决于数据集
                std=[0.5, 0.5, 0.5]
           )
        ]
    )

    dataset = SignboardDataset(image_root, cls_label_path, transform)

    device = paddle.set_device("cpu")
    data_loader = DataLoader(dataset, 
                            batch_size=10, 
                            places=device,
                            shuffle=False, 
                            drop_last=True,
                            return_list=True,
                            use_shared_memory=True)

    for batch_id, data in enumerate(data_loader()):
        print(data)


        
