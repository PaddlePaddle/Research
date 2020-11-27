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
        self.datafile = 'data/omniglot/data/'
        print('loading omniglot dataset from {} ......'.format(self.datafile))
        if self.mode == 'train':
            character_folders = [[os.path.join(self.datafile, family, character)] \
                    for family in os.listdir(self.datafile) \
                    if os.path.isdir(os.path.join(self.datafile, family)) \
                    for character in os.listdir(os.path.join(self.datafile, family))]
            random.shuffle(character_folders)
            character_folders_train = character_folders[:1200]
            character_folders_val = character_folders[1200:1300]
            character_folders_test = character_folders[1300:-1]
            with open('data/omniglot/train.csv','w') as file:
                wr = csv.writer(file)
                wr.writerows(character_folders_train)
            with open('data/omniglot/val.csv','w') as file:
                wr = csv.writer(file)
                wr.writerows(character_folders_val)
            with open('data/omniglot/test.csv','w') as file:
                wr = csv.writer(file)
                wr.writerows(character_folders_test) 

        csv_path = 'data/omniglot/' + self.mode + '.csv'
        lines = [x.strip() for x in open(csv_path, 'r').readlines()]
        data = []
        labels = []
        rotate_angles = [0,90,180,270]
        label_index = 0
        for x in lines:
            class_names = glob.glob(os.path.join(x, '*'))
            for angle in rotate_angles:
                class_imgs = [np.asarray(Image.open(file).rotate(angle)) for file in class_names]
                class_labels = [label_index for _ in range(len(class_imgs))]
                data += class_imgs
                labels += class_labels
                label_index += 1
        self.n_classes = label_index
        self.data = data
        self.label = np.array(labels)
        print('omniglot dataset {} load done'.format(self.mode))

    def __getitem__(self, index):
        images = np.stack([self.data[i] for i in index])
        label = np.stack([self.label[i] for i in index])
        imgs = np.expand_dims(images, axis=1)
        return imgs, label




