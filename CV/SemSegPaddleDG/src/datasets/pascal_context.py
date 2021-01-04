from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import paddle
import numpy as np

from PIL import Image
from .base import BaseDataSet

__all__ = ['pascal_context_train', 'pascal_context_qucik_val', 'pascal_context_eval']

#  globals
context_data_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
context_data_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


class Pascal_Context(BaseDataSet):
    """prepare Pascal_Context path_pairs"""

    NUM_CLASS = 59

    def __init__(self, root='./dataset', split='train', **kwargs):
        super(Pascal_Context, self).__init__(root, split, **kwargs)
        if os.sep == '\\':  # windows
            root = root.replace('/', '\\')
        assert os.path.exists(root), "please download pascal_context data_set, put in dataset(dir),or check root"
        self.image_path, self.label_path = self._get_pascal_context_pairs(root, split)
        assert len(self.image_path) == len(self.label_path), "please check image_length = label_length"
        self.print_param()

    def print_param(self):  # 用于核对当前数据集的信息
        print('INFO: dataset_root: {}, split: {}, '
              'base_size: {}, crop_size: {}, scale: {}, '
              'image_length: {}, label_length: {}'.format(self.root, self.split, self.base_size,
                                                          self.crop_size, self.scale, len(self.image_path),
                                                          len(self.label_path)))

    @staticmethod
    def _get_pascal_context_pairs(root, split):

        def get_pairs(root, file_txt, label_dir='GroundTruth_trainval_png', img_dir='JPEGImages'):
            file_path = os.path.join(root, file_txt)
            with open(file_path, 'r') as f:
                file_list_item = f.readlines()

            image_dir = os.path.join(root, img_dir)
            image_path = [os.path.join(image_dir, x.strip() + '.jpg') for x in file_list_item]

            gt_dir = os.path.join(root, label_dir)
            label_path = [os.path.join(gt_dir, x.strip() + '.png') for x in file_list_item]

            return image_path, label_path

        if split == 'train':
            image_path, label_path = get_pairs(root, 'ImageSets/train.txt')
        elif split == 'val':
            image_path, label_path = get_pairs(root, 'ImageSets/val.txt')
        elif split == 'test':
            image_path, label_path = get_pairs(root, 'ImageSets/test.txt')  # 返回文件路径，test_label并不存在
        else:  # 'train_val'
            image_path1, label_path1 = get_pairs(root, 'ImageSets/train.txt')
            image_path2, label_path2 = get_pairs(root, 'ImageSets/val.txt')
            image_path, label_path = image_path1 + image_path2, label_path1 + label_path2
        return image_path, label_path

    def sync_transform(self, image, label, aug=True):
        crop_size = self.crop_size
        if self.scale:
            short_size = random.randint(int(self.base_size * 0.75), int(self.base_size * 2.0))
        else:
            short_size = self.base_size

        # 随机左右翻转
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = image.size

        # 同比例缩放
        if h > w:
            out_w = short_size
            out_h = int(1.0 * h / w * out_w)
        else:
            out_h = short_size
            out_w = int(1.0 * w / h * out_h)
        image = image.resize((out_w, out_h), Image.BILINEAR)
        label = label.resize((out_w, out_h), Image.NEAREST)

        # deg = random.uniform(-10, 10)
        # image = image.rotate(deg, resample=Image.BILINEAR)
        # label = label.rotate(deg, resample=Image.NEAREST)
        # 四周填充
        if short_size < crop_size:
            pad_h = crop_size - out_h if out_h < crop_size else 0
            pad_w = crop_size - out_w if out_w < crop_size else 0
            image = ImageOps.expand(image, border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                                    fill=0)
            label = ImageOps.expand(label, border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
                                    fill=0)

        # 随机裁剪
        w, h = image.size
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        image = image.crop((x, y, x + crop_size, y + crop_size))
        label = label.crop((x, y, x + crop_size, y + crop_size))

        if aug:
            # 高斯模糊，可选
            if random.random() > 0.7:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

            # 可选
            if random.random() > 0.7:
                # 随机亮度
                factor = np.random.uniform(0.75, 1.25)
                image = ImageEnhance.Brightness(image).enhance(factor)

                # 颜色抖动
                factor = np.random.uniform(0.75, 1.25)
                image = ImageEnhance.Color(image).enhance(factor)

                # 随机对比度
                factor = np.random.uniform(0.75, 1.25)
                image = ImageEnhance.Contrast(image).enhance(factor)

                # 随机锐度
                factor = np.random.uniform(0.75, 1.25)
                image = ImageEnhance.Sharpness(image).enhance(factor)

        return image, label

    def sync_val_transform(self, image, label):
        crop_size = self.crop_size
        short_size = self.base_size

        w, h = image.size

        # # 同比例缩放
        if h > w:
            out_w = short_size
            out_h = int(1.0 * h / w * out_w)
        else:
            out_h = short_size
            out_w = int(1.0 * w / h * out_h)
        image = image.resize((out_w, out_h), Image.BILINEAR)
        label = label.resize((out_w, out_h), Image.NEAREST)

        # # 中心裁剪
        w, h = image.size
        x1 = int(round((w - crop_size) / 2.))
        y1 = int(round((h - crop_size) / 2.))
        image = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        label = label.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        return image, label

    def get_path_pairs(self):
        return self.image_path, self.label_path



def context_mapper_train(sample):
    image_path, label_path, context = sample
    image = Image.open(image_path, mode='r').convert('RGB')
    label = Image.open(label_path, mode='r')
    
    image, label = context.sync_transform(image, label)
    image_array = np.array(image)  # HWC
    label_array = np.array(label)  # HW
    label_array = label_array - 1 

    height, width = label_array.shape

    image_array = image_array.transpose((2, 0, 1))  # CHWA
    image_array = image_array / 255.0
    image_array = (image_array - context_data_mean) / context_data_std
    image_array = image_array.astype('float32')
    label_array = label_array.astype('int64')
    return image_array, label_array


def context_mapper_val(sample):
    image_path, label_path, context = sample
    image = Image.open(image_path, mode='r').convert('RGB')
    label = Image.open(label_path, mode='r')

    image, label = context.sync_val_transform(image, label)
    image_array = np.array(image)  # HWC
    label_array = np.array(label)  # HW
    label_array = label_array - 1 

    image_array = image_array.transpose((2, 0, 1))  # CHW
    image_array = image_array / 255.0
    image_array = (image_array - context_data_mean) / context_data_std
    image_array = image_array.astype('float32')
    label_array = label_array.astype('int64')
    return image_array, label_array, image_path


def context_mapper_test(sample):
    image_path, label_path = sample  # label is path
    print(image_path, label_path)
    image = Image.open(image_path, mode='r').convert('RGB')
    label = Image.open(label_path, mode='r')
    label_array = np.array(label)
    label_array = label_array - 1
    image_array = image
    return image_array, label_array, label_path  # image is a picture, label is path


def pascal_context_train(root='.', base_size=520, crop_size=520, scale=True, xmap=True, batch_size=1, gpu_num=1):
    context = Pascal_Context(root=root, split='train', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = context.get_path_pairs()

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
            yield image_path[i], label_path[i], context
    if xmap:
        return paddle.reader.xmap_readers(context_mapper_train, reader, 4, 32)
    else:
        return paddle.reader.map_readers(context_mapper_train, reader)


def pascal_context_quick_val(root='./dataset', base_size=520, crop_size=520, scale=True, xmap=True, batch_size=1, gpu_num=1):
    context = Pascal_Context(root=root, split='val', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = context.get_path_pairs()

    def reader():
        if len(image_path) % (batch_size * gpu_num) != 0:
            length = (len(image_path) // (batch_size * gpu_num)) * (batch_size * gpu_num)
        else:
            length = len(image_path)
        for i in range(length):
            yield image_path[i], label_path[i], context

    if xmap:
        return paddle.reader.xmap_readers(context_mapper_val, reader, 4, 32)
    else:
        return paddle.reader.map_readers(context_mapper_val, reader)


def pascal_context_eval(root='./dataset', base_size=520, crop_size=520, scale=True, xmap=True, batch_size=1, gpu_num=1):
    context = Pascal_Context(root=root, split='val', base_size=base_size, crop_size=crop_size, scale=scale)
    image_path, label_path = context.get_path_pairs()

    def reader():
        for i in range(len(image_path)):
            yield image_path[i], label_path[i]
    if xmap:
        return paddle.reader.xmap_readers(context_mapper_test, reader, 4, 32)
    else:
        return paddle.reader.map_readers(context_mapper_test, reader)
