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
        self.datafile = 'data/cub/CUB_images/'
        print('loading cub dataset from {} ......'.format(self.datafile))
        csv_path = 'data/cub/' + self.mode + '.csv'
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = []
        class_name_all = []
        class_name_unique = set()
        for x in lines:
            file,class_name = x.split(',')
            data.append(os.path.join(self.datafile+file))
            class_name_all.append(class_name)
            class_name_unique.add(class_name)
        self.n_classes = len(class_name_unique)
        name_label_dic = dict(zip(sorted(list(class_name_unique)), list(range(self.n_classes))))
        label = [name_label_dic[x] for x in class_name_all]
        self.data = [np.array(Image.open(file)) for file in data]
        self.label = np.array(label)
        print('cub dataset {} load done'.format(self.mode))

    def __getitem__(self, index):
        # img_files, label = self.data[index], self.label[index]
        # images = np.stack([np.array(Image.open(file)) for file in img_files])
        images = np.stack([self.data[i] for i in index])
        label = np.stack([self.label[i] for i in index])
        imgs = np.transpose(images, [0,3,1,2])
        return imgs, label




