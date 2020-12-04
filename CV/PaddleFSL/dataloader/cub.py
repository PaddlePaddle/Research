import numpy as np
import paddle
import glob
import os
from PIL import Image
import paddle.fluid as fluid
import pickle as pkl

class LoadData():
    def __init__(self,mode):
        self.mode = mode
        self.datafile = 'data/cub/cub_' + self.mode + '.pkl'
        print('loading cub dataset from {} ......'.format(self.datafile))
        with open(self.datafile, "rb") as f:
            data = pkl.load(f)
            self.data = data['data'] 
            self.label = data['label']
        print('cub dataset {} load done'.format(self.mode))

    def __getitem__(self, index):
        # img_files, label = self.data[index], self.label[index]
        # images = np.stack([np.array(Image.open(file)) for file in img_files])
        images = np.stack([self.data[i] for i in index])
        label = np.stack([self.label[i] for i in index])
        imgs = np.transpose(images, [0,3,1,2])
        return imgs, label




