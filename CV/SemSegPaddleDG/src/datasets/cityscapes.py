from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import paddle
import numpy as np
from PIL import Image
from .base import BaseDataSet

__all__ = ['cityscapes_train', 'cityscapes_train_val', 'cityscapes_test', 'cityscapes_train_val', 'cityscapes_eval']

#  globals
city_data_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
city_data_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


class Cityscapes(BaseDataSet):
    """prepare cityscapes path_pairs"""
    BASE_DIR = 'cityscapes_old'
    NUM_CLASS = 19

    def __init__(self, root='./dataset', split='train', **kwargs):
        super(Cityscapes, self).__init__(root, split, **kwargs)
        if os.sep == '\\':  # windows
            root = root.replace('/', '\\')
        if split in ['train', 'val', 'eval']:
            root = os.path.join(root, 'cityscapes') # ../cityscapes
        else:
            root = os.path.join(root, 'cityscapes_old')                                                                                                                                                                                                                                       
        assert os.path.exists(root), "please download cityscapes data_set, put in dataset(dir),or check root"
        self.image_path, self.label_path = self._get_cityscapes_pairs(root, split)
        assert len(self.image_path) == len(self.label_path), "please check image_length = label_length"
        self.print_param()

    def print_param(self):  # 用于核对当前数据集的信息
        print('INFO: dataset_root: {}, split: {}, '
              'base_size: {}, crop_size: {}, scale: {}, '
              'image_length: {}, label_length: {}'.format(self.root, self.split, self.base_size,
                                                          self.crop_size, self.scale, len(self.image_path),
                                                          len(self.label_path)))

    @staticmethod
    def _get_cityscapes_pairs(root, split):

        def get_pairs(root, file_image, file_label):
            file_image = os.path.join(root, file_image)
            file_label = os.path.join(root, file_label)
            with open(file_image, 'r') as f:
                file_list_image = f.read().split()
            with open(file_label, 'r') as f:
                file_list_label = f.read().split()
            if os.sep == '\\':  # for windows
                image_path = [os.path.join(root, x.replace('/', '\\')) for x in file_list_image]
                label_path = [os.path.join(root, x.replace('/', '\\')) for x in file_list_label]
            else:
                image_path = [os.path.join(root, x) for x in file_list_image]
                label_path = [os.path.join(root, x) for x in file_list_label]
            return image_path, label_path
        
        def _get_pairs(root, path):
            file_path = os.path.join(root, path)
            assert file_path[-3:] == 'lst', 'input file is not lst format, please check usage!'
            with open(file_path, 'r') as f:
                file_list = f.readlines()
            image_path, label_path = [], []
            if os.sep == '\\': # for windows
                raise ValueError('windows is not supported by now')
            else:
                for sample in file_list:
                    image_path.append(os.path.join(root, sample.strip().split('\t')[0]))
                    label_path.append(os.path.join(root, sample.strip().split('\t')[1]))
            return image_path, label_path

        if split == 'train':
            image_path, label_path = _get_pairs(root, 'cityscapes_list/train.lst')
        elif split == 'val':
            image_path, label_path = _get_pairs(root, 'cityscapes_list/val.lst')
        elif split == 'eval':
            image_path, label_path = _get_pairs(root, 'cityscapes_list/val.lst')
        else:  # 'train_val'
            image_path1, label_path1 = _get_pairs(root, 'cityscapes_list/train.lst')
            image_path2, label_path2 = _get_pairs(root, 'cityscapes_list/val.lst')
            image_path, label_path = image_path1+image_path2, label_path1+label_path2
        return image_path, label_path

    def get_path_pairs(self):
        return self.image_path, self.label_path
                                                  

def city_mapper_train(sample):
    image_path, label_path, city = sample
    image = Image.open(image_path, mode='r').convert('RGB')
    label = Image.open(label_path, mode='r')
    
    image, label = city.sync_transform(image, label)
    image_array = np.array(image)  # HWC
    label_array = np.array(label)  # HW

    height, width = label_array.shape

    image_array = image_array.transpose((2, 0, 1))  # CHWA
    image_array = image_array / 255.0
    image_array = (image_array - city_data_mean) / city_data_std
    image_array = image_array.astype('float32')
    label_array = label_array.astype('int64')
    # label_map (33) to class_map (19)
    gt_h = label_array.shape[0]
    gt_w = label_array.shape[1]
    id_to_trainid = [255, 255, 255, 255, 255,
                             255, 255, 255, 0, 1,
                             255, 255, 2, 3, 4,
                             255, 255, 255, 5, 255,
                             6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15,
                             255, 255, 16, 17, 18]
    gt = np.zeros([gt_h, gt_w])
    for h in range(gt_h):
        for w in range(gt_w):
                    gt[h][w] = id_to_trainid[int(label_array[h][w])+1]
    return image_array, gt


def city_mapper_val(sample):
    image_path, label_path, city = sample
    image = Image.open(image_path, mode='r').convert('RGB')
    label = Image.open(label_path, mode='r')

    image, label = city.sync_val_transform(image, label)
    image_array = np.array(image)  # HWC
    label_array = np.array(label)  # HW

    image_array = image_array.transpose((2, 0, 1))  # CHW
    image_array = image_array / 255.0
    image_array = (image_array - city_data_mean) / city_data_std
    image_array = image_array.astype('float32')
    label_array = label_array.astype('int64')
    # label_map (33) to class_map (19)
    gt_h = label_array.shape[0]
    gt_w = label_array.shape[1]
    id_to_trainid = [255, 255, 255, 255, 255,
                             255, 255, 255, 0, 1,
                             255, 255, 2, 3, 4,
                             255, 255, 255, 5, 255,
                             6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15,
                             255, 255, 16, 17, 18]
    gt = np.zeros([gt_h, gt_w])
    for h in range(gt_h):
        for w in range(gt_w):
                    gt[h][w] = id_to_trainid[int(label_array[h][w])+1]

    return image_array, gt, image_path


def city_mapper_test(sample):
    image_path, label_path = sample  # label is path
    print(image_path, label_path)
    image = Image.open(image_path, mode='r').convert('RGB')
    label = Image.open(label_path, mode='r')
    label_array = np.array(label)
    image_array = image
    # label_map (33) to class_map (19)
    gt_h = label_array.shape[0]
    gt_w = label_array.shape[1]
    id_to_trainid = [255, 255, 255, 255, 255,
                             255, 255, 255, 0, 1,
                             255, 255, 2, 3, 4,
                             255, 255, 255, 5, 255,
                             6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15,
                             255, 255, 16, 17, 18]
    gt = np.zeros([gt_h, gt_w])
    for h in range(gt_h):
        for w in range(gt_w):
                    gt[h][w] = id_to_trainid[int(label_array[h][w])+1]
    #return image_array, label_array, label_path  # image is a picture, label is path
    return image_array, gt, label_path  # image is a picture, label is path


# root, base_size, crop_size; gpu_num必须设置，否则syncBN会出现某些卡没有数据的情况
def cityscapes_train(root='./dataset', base_size=1024, crop_size=768, scale=True, xmap=True, batch_size=1, gpu_num=1):
    city = Cityscapes(root=root, split='train', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = city.get_path_pairs()

    def reader():
        if len(image_path) % (batch_size * gpu_num) != 0:
            length = (len(image_path) // (batch_size * gpu_num)) * (batch_size * gpu_num)
        else:
            length = len(image_path)
        for i in range(length):
            if i == 0:
                cc = list(zip(image_path, label_path))
                random.shuffle(cc)
                image_path[:], label_path[:] = zip(*cc)
            yield image_path[i], label_path[i], city
    if xmap:
        return paddle.reader.xmap_readers(city_mapper_train, reader, 4, 32)
    else:
        return paddle.reader.map_readers(city_mapper_train, reader)


def cityscapes_quick_val(root='./dataset', base_size=1024, crop_size=768, scale=True, xmap=True, batch_size=1, gpu_num=1):
    city = Cityscapes(root=root, split='val', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = city.get_path_pairs()

    def reader():
        if len(image_path) % (batch_size * gpu_num) != 0:
            length = (len(image_path) // (batch_size * gpu_num)) * (batch_size * gpu_num)
        else:
            length = len(image_path)
        for i in range(length):
            yield image_path[i], label_path[i], city

    if xmap:
        return paddle.reader.xmap_readers(city_mapper_val, reader, 4, 32)
    else:
        return paddle.reader.map_readers(city_mapper_val, reader)


def cityscapes_train_val(root='./dataset', base_size=1024, crop_size=768, scale=True, xmap=True, batch_size=1, gpu_num=1):
    city = Cityscapes(root=root, split='train_val', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = city.get_path_pairs()

    def reader():
        if len(image_path) % (batch_size * gpu_num) != 0:
            length = (len(image_path) // (batch_size * gpu_num)) * (batch_size * gpu_num)
        else:
            length = len(image_path)
        for i in range(length):
            if i == 0:
                cc = list(zip(image_path, label_path))
                random.shuffle(cc)
                image_path[:], label_path[:] = zip(*cc)
            yield image_path[i], label_path[i], city

    if xmap:
        return paddle.reader.xmap_readers(city_mapper_train, reader, 4, 32)
    else:
        return paddle.reader.map_readers(city_mapper_train, reader)


def cityscapes_test(root='./dataset', base_size=2048, crop_size=1024, scale=True, xmap=True, batch_size=1, gpu_num=1):
    # 实际未使用base_size, crop_size, scale
    city = Cityscapes(root=root, split='test', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = city.get_path_pairs()

    def reader():
        for i in range(len(image_path)):
            yield image_path[i], label_path[i]
    if xmap:
        return paddle.reader.xmap_readers(city_mapper_test, reader, 4, 32)
    else:
        return paddle.reader.map_readers(city_mapper_test, reader)


def cityscapes_eval(root='./dataset', base_size=2048, crop_size=1024, scale=True, xmap=True, batch_size=1, gpu_num=1):
    # 实际未使用base_size, crop_size, scale
    city = Cityscapes(root=root, split='eval', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = city.get_path_pairs()

    def reader():
        for i in range(len(image_path)):
            yield image_path[i], label_path[i]
    if xmap:
        return paddle.reader.xmap_readers(city_mapper_test, reader, 4, 32)
    else:
        return paddle.reader.map_readers(city_mapper_test, reader)

