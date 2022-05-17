# -*-coding utf-8 -*-
##########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
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
#
##########################################################################
"""
数据变换器
"""

import numpy as np
import numbers
import collections
import random
import math

import cv2

from . import functional as F
from easymia.core.abstract_transforms import AbstractTransform
from easymia.libs import manager

@manager.TRANSFORMS.add_component
class Compose(AbstractTransform):
    """
    Do transformation on input data with corresponding pre-processing and augmentation operations.
    The shape of input data to all operations is [height, width, channels].
    Args:
        transforms (list): A list contains data pre-processing or augmentation. Empty list means only reading images, no transformation.
        to_rgb (bool, optional): If converting image to RGB color space. Default: True.
    Raises:
        TypeError: When 'transforms' is not a list.
        ValueError: when the length of 'transforms' is less than 1.
    """

    def __init__(self, mode, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        super().__init__(mode)

    def __clas__(self, im):
        """
        Args:
            im (np.ndarray): It is either image path or image object.
        Returns:
            (np.array). Image after transformation.
        """
        for op in self.transforms:
            im = op(im)
        return im


@manager.TRANSFORMS.add_component
class RandomHorizontalFlip(AbstractTransform):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, mode, prob=0.5):
        """
        init
        """
        self.prob = prob
        super().__init__(mode)

    def __clas__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """

        if random.random() < self.prob:
            return F.hflip(img)
        return img
        

@manager.TRANSFORMS.add_component
class RandomVerticalFlip(AbstractTransform):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, mode, prob=0.5):
        """
        init
        """
        self.prob = prob
        super().__init__(mode)

    def __clas__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        if random.random() < self.prob:
            return F.vflip(img)
        return img


@manager.TRANSFORMS.add_component
class RandomResizedCrop(AbstractTransform):
    """Crop the given numpy ndarray to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: cv2.INTER_CUBIC
    """

    def __init__(self, mode, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_CUBIC):
        """
        init
        """
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

        super().__init__(mode)

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        params_ret = collections.namedtuple('params_ret', ['i', 'j', 'h', 'w'])
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return params_ret(i, j, h, w)

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return params_ret(i, j, w, w)

    def __clas__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped and resized.
        Returns:
            numpy ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


@manager.TRANSFORMS.add_component
class RandomRotation(AbstractTransform):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, mode, degrees, center=None):
        """
        init
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.center = center

        super().__init__(mode)

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __clas__(self, img):
        """
            img (numpy ndarray): Image to be rotated.
        Returns:
            numpy ndarray: Rotated image.
        """

        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.center)



@manager.TRANSFORMS.add_component
class Resize(AbstractTransform):
    """Resize the input numpy ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``, bicubic interpolation
    """

    def __init__(self, mode, size, interpolation=cv2.INTER_LINEAR):
        """
        resize
        """
        # assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, collections.abc.Iterable) and len(size) == 2:
            if type(size) == list:
                size = tuple(size)
            self.size = size
        else:
            raise ValueError('Unknown inputs for size: {}'.format(size))
        self.interpolation = interpolation
        super().__init__(mode)

    def __clas__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)