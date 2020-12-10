import numpy as np
import paddle
import paddle.fluid as fluid
import pickle as pkl

class LoadData():
    def __init__(self,mode):
        self.mode = mode
        self.datafile = 'data/CIFAR-FS/CIFAR_FS_' + self.mode + '.pickle'
        print('loading CIFAR-FS dataset from {} ......'.format(self.datafile))
        with open(self.datafile, 'rb') as f:
            images_labels = pkl.load(f, encoding='latin1')  
        data = images_labels['data']   # [36000,32,32,3] , random order, datatype = unit8
        self.data = np.transpose(data, [0,3,1,2])  # [36000,3,32,32]
        self.label = np.array(images_labels['labels'])   # [36000], int
        print('CIFAR-FS dataset load done')

    def __getitem__(self, index):
        return self.data[index], self.label[index]




