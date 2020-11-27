import numpy as np
import paddle
import glob
import os
from PIL import Image
import paddle.fluid as fluid
import pickle as pkl
import cv2
from tqdm import tqdm

class LoadData():
    def __init__(self,mode):
        self.mode = mode
        self.datafile = 'data/tiered-imagenet/data/tiered-imagenet/' + self.mode + '_images_png.pkl'
        self.labelfile = 'data/tiered-imagenet/data/tiered-imagenet/' + self.mode + '_labels.pkl'
        print('loading tiered-imagenet dataset from {} ......'.format(self.datafile))
        with open(self.datafile, 'rb') as f:
            images_raw = pkl.load(f)                # [448695--[n]]
        with open(self.labelfile, 'rb') as f:
            labels_all = pkl.load(f)
            labels = labels_all["label_specific"] # [448695]
        images_data = np.zeros([len(images_raw), 84, 84, 3], dtype=np.uint8)
        for ii, item in tqdm(enumerate(images_raw), desc='decompress'):
            im = cv2.imdecode(item, 1)
            images_data[ii] = im
        self.data = images_data
        self.label = labels
        print('tiered-imagenet dataset load done')

    def __getitem__(self, index):
        # img_files, label = self.data[index], self.label[index]
        # images = np.stack([np.array(Image.open(file)) for file in img_files])
        images = np.stack([self.data[i] for i in index])
        label = np.stack([self.label[i] for i in index])
        imgs = np.transpose(images, [0,3,1,2])
        return imgs, label




