import numpy as np
import paddle
import paddle.fluid as fluid
import pickle as pkl
from PIL import Image
import os
import glob
import csv
import random

class LoadData():
    def __init__(self,mode):
        self.mode = mode
        self.datafile = 'data/omniglot/omniglot_' + self.mode + '.pkl'
        print('loading omniglot dataset from {} ......'.format(self.datafile))
        with open(self.datafile, "rb") as f:
            data = pkl.load(f)
            self.data = data['data'] 
            self.label = data['label']
        print('omniglot dataset {} load done'.format(self.mode))

    def __getitem__(self, index):
        images = np.stack([self.data[i] for i in index])
        label = np.stack([self.label[i] for i in index])
        imgs = np.expand_dims(images, axis=1)
        return imgs, label




