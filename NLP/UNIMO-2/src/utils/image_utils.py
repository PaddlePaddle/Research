#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File: img2txt_reader.py
Author: liwei(liwei85@baidu.com)
Date: 2021-06-21 19:27
Desc: image file reader and normalization
"""

import cv2
import numpy as np
import random
import paddle
from PIL import Image, ImageFile, UnidentifiedImageError
from multiprocessing.pool import ThreadPool

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageReader(object):
    def __init__(self,
                 target_shape_h=224,
                 target_shape_w=224,
                 data_format="channels_last",
                 dtype='float32',
                 mean_value=None,
                 std_value=None,
                 with_random_crop=False,
                 with_random_flip=False):
        if std_value is None:
            std_value = [0.5, 0.5, 0.5]
        if mean_value is None:
            mean_value = [0.5, 0.5, 0.5]

        if target_shape_h and target_shape_w:
            self.target_size = (target_shape_h, target_shape_w)
        else:
            self.target_size = None
        self.data_format = data_format
        self.dtype = dtype
        assert data_format in {'channels_first', 'channels_last'}

        self.with_random_crop = with_random_crop
        self.with_random_clip = with_random_flip

        self.mean = np.array(mean_value).astype(self.dtype)
        self.std = np.array(std_value).astype(self.dtype)

    def _load_img(self, path):
        '''
        :param path: img path
        :return: PIL image, (w, h)
        '''
        try:
            img = Image.open(path)  # (w, h)
        except UnidentifiedImageError:
            print("UnidentifiedImageError: ", path)
            return None
        except:
            print("Image.open Fail: ", path)
            img = cv2.imread(path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.with_random_crop:
            img = self.random_crop(img, self.target_size)

        if self.with_random_clip:
            img = self.random_flip(img)

        if self.target_size:
            if img.size != self.target_size:
                img = img.resize(self.target_size)
        return img

    def _img_to_array(self, img):
        '''
        :param img: PIL image, (w, h)
        :return: numpy array, (c, h, w) or (h, w, c)
        '''
        x = np.asarray(img, dtype=self.dtype)  # (h, w, c)
        assert len(x.shape) == 3
        if self.data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
        return x

    def process_image(self, image_path):
        """Given an image, process it and return the array."""
        # Load the image.
        image = self._load_img(image_path)
        if image is None:
            return None
        img_arr = self._img_to_array(image)
        x = (img_arr / 255.).astype('float32')  #
        x = (x - self.mean) / self.std  # normalize images
        return x

    def random_crop(self, img, target_size, p=0.5):
        """random crop image"""
        v1 = random.random()
        if v1 < p:
            return img

        w, h = img.size
        th, tw = target_size
        if w > tw and h > th:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img.crop((x1, y1, x1 + tw, y1 + th))
        elif w > tw:
            x1 = np.random.randint(0, w - tw)
            return img.crop((x1, 0, x1 + tw, h))
        elif h > th:
            y1 = random.randint(0, h - th)
            return img.crop((0, y1, w, y1 + th))
        else:
            return img

    def random_flip(self, img, p=0.5):
        """
        :param method: One of :py:data:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:data:`PIL.Image.FLIP_TOP_BOTTOM`, :py:data:`PIL.Image.ROTATE_90`,
          :py:data:`PIL.Image.ROTATE_180`, :py:data:`PIL.Image.ROTATE_270`.
        """
        v1 = random.random()
        if v1 < p:
            v2 = random.random()
            if v2 < 0.2:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            elif v2 < 0.4:
                return img.transpose(Image.FLIP_TOP_BOTTOM)
            elif v2 < 0.6:
                return img.transpose(Image.ROTATE_90)
            elif v2 < 0.8:
                return img.transpose(Image.ROTATE_180)
            else:
                return img.transpose(Image.ROTATE_270)
        else:
            return img


def process_image(image_id,
                  image_path,
                  target_shape_h=224,
                  target_shape_w=224,
                  data_format="channels_last",
                  dtype='float32',
                  with_random_crop=False,
                  with_random_flip=False):
    img_reader = ImageReader(target_shape_h=target_shape_h,
                             target_shape_w=target_shape_w,
                             data_format=data_format,
                             dtype=dtype,
                             with_random_crop=with_random_crop,
                             with_random_flip=with_random_flip)
    x = img_reader.process_image(image_path)
    return (image_id, x)


def process_image_with_multi_proc(images_list, proc_num, func):
    pool = ThreadPool(processes=proc_num)
    thread_list = []
    for image_id, img_path in images_list:
        out = pool.apply_async(func=func, args=(image_id, img_path,))
        thread_list.append(out)
    pool.close()
    pool.join()
    results = []
    for p in thread_list:
        result = p.get()
        results.append(result)
    return results


def test_img_reader():
    img_path = "999999"
    img_reader = ImageReader()
    x = img_reader.process_image(img_path)
    print(x.shape, x)


def test_process_image():
    img_path = "999999"
    x = process_image(1, img_path)
    print(x[1].shape, x)


def test_process_image_with_multi_proc(img_paths):
    img_arrs = process_image_with_multi_proc(img_paths, 3, process_image)
    for i in range(len(img_arrs)):
        x = img_arrs[i]
        print(x[1].shape, x)
        h, w, c = x[1].shape
        print(x[1].reshape(h * w // (16 * 16), -1).shape)
        print(h, w, c)


def parse_line(line):
    """ parse one line to token_ids, sentence_ids, pos_ids, label
    """
    line = line.strip('\r\n').split(";")
    (image_id, image_path, caption_id, token_ids, sent_ids, pos_ids, seg_labels, label, image_label) = line
    return image_id, image_path


def read_file(file):
    image_paths = []
    with open(file, "r") as f:
        for line in f:
            img_id, img_path = parse_line(line)
            image_paths.append((img_id, img_path))

            if len(image_paths) == 100:
                process_image_with_multi_proc(image_paths, 20, process_image)
                image_paths = []


if __name__ == "__main__":
    img_filelist = "data/images-prompt/image_filelist.imagenet21k_openimages"
    img_files = open(img_filelist).readlines()
    for img_file in img_files:
        img_file = img_file.strip()
        print(img_file)
        read_file(img_file)
