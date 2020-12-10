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
        self.datafile = 'data/mini-imagenet/mini-imagenet-cache-' + self.mode + '.pkl'
        print('loading miniimagenet dataset from {} ......'.format(self.datafile))
        with open(self.datafile, "rb") as f:
            data = pkl.load(f, encoding='bytes')
            image_data = data['image_data']
            class_dict = data['class_dict']
        label = np.zeros([image_data.shape[0]], dtype=int)
        for i,key in enumerate(sorted(class_dict.keys())):
            label[np.asarray(class_dict[key])] = i
        self.data = image_data
        self.label = label
        print('miniimagenet dataset load done')

    def __getitem__(self, index):
        # img_files, label = self.data[index], self.label[index]
        # images = np.stack([np.array(Image.open(file)) for file in img_files])
        images = np.stack([self.data[i] for i in index])
        label = np.stack([self.label[i] for i in index])
        imgs = np.transpose(images, [0,3,1,2])
        return imgs, label




