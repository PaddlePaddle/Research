# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: main.py
func: 训练启动代码 
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/06/11
"""

import math
import random

import numpy as np
import cv2
from PIL import Image


class RandomPerspective(object):
    """-定概率进行随机透视变换"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        img = np.array(img)
        img = img[:, :, ::-1]

        img_h, img_w = img.shape[:2]
        thresh = img_h // 3
        src_pts = list()
        dst_pts = list()
        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])
        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])

        M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
        img = cv2.warpPerspective(img, M, (img_w, img_h)) 
        img = Image.fromarray(img[:, :, ::-1])

        return img


class RandomRotate90(object):
    """一定概率旋转90度
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """call"""
        if random.uniform(0, 1) >= self.p:
            return img
        img = np.array(img)
        if random.choice([0, 1]) == 1:
            img = np.rot90(img)
        else:
            img = np.rot90(img, -1)

        img = Image.fromarray(img)

        return img


class RandomHSV(object):
    """一定概率HSV变化
    Args:
        p: 概率
        h: hue值变化范围
        s: 饱和度变化范围
        v: 明度变化范围
    """
    def __init__(self, p, h, s, v):
        self.p = p
        self.h = h
        self.s = s
        self.v = v

    def __call__(self, img):
        """call"""
        if random.uniform(0, 1) >= self.p:
            return img
        img = np.array(img)
        img = img[:, :, ::-1]

        hue_delta = np.random.randint(-self.h, self.h)
        sat_mult = 1 + np.random.uniform(-self.s, self.s)
        val_mult = 1 + np.random.uniform(-self.v, self.v)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255

        img = cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = Image.fromarray(img[:, :, ::-1])

        return img


class RandomRotation(object):
    """一定概率旋转
    """
    def __init__(self, p, angle):
        self.p = p
        self.angle = angle

    def __call__(self, img):
        if random.uniform(0, 1) >= self.p:
            return img
        img = np.array(img)
        img = img[:, :, ::-1]

        angle=np.random.randint(-self.angle, self.angle)

        (h, w) = img.shape[:2] #2
        center = (w // 2, h // 2) #4
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0) #5
        img = cv2.warpAffine(img, M, (w, h)) #6
        img = Image.fromarray(img[:, :, ::-1])

        return img #7


class RandomCropResize(object):
    """给定crop比例范围返回对应的图像
    """
    def __init__(self, ratios=[0.8, 1], size=[224, 224], method='random'):
        self.ratios = ratios
        self.size = size
        self.method = method

    def get_lefttop(self, crop_size, img_size):
        """get left-top point"""
        if self.method == 'center':
            return [(img_size[0] - crop_size[0]) // 2, (img_size[1] - crop_size[1]) // 2]
        elif self.method == 'random':
            x = random.randint(0, img_size[0] - crop_size[0])
            y = random.randint(0, img_size[1] - crop_size[1])
            return [x, y]
        elif self.method == 'grid':
            grid_x = random.randint(0, self.grid[0] - 1)
            grid_y = random.randint(0, self.grid[1] - 1)
            x = grid_x * ((img_size[0] - crop_size[0]) // (self.grid[0] - 1))
            y = grid_y * ((img_size[1] - crop_size[1]) // (self.grid[1] - 1))
            return [x, y]
        else:
            raise NotImplementedError('Random Crop Method {} Undefined!'.format(self.method))

    def __call__(self, img):
        """
        Args:
            img (Image):   Image to be cropped.
        Returns:
            Image:  Cropped image.
        """
        img = np.array(img)
        height, width, _ = img.shape
        ratio = random.uniform(self.ratios[0], self.ratios[1])
        target_size = [int(width * ratio), int(height * ratio)]
        offset_left, offset_up = self.get_lefttop(target_size, [width, height])
        img = img[offset_up:offset_up + target_size[1], offset_left:offset_left + target_size[0]]
        img = cv2.resize(img, tuple(self.size), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(img)

        return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        """call"""

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img