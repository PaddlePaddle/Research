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
数据变换function
"""

import collections
import numbers
import random

import numpy as np
import cv2
import scipy
import scipy.ndimage
import SimpleITK as sitk

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3, 4})


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        numpy ndarray: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return img[i:i + h, j:j + w]


def hflip(img):
    """Horizontally flip the given numpy ndarray.
    Args:
        img (numpy ndarray): image to be flipped.
    Returns:
        numpy ndarray:  Horizontally flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    # img[:,::-1] is much faster, but doesn't work with torch.from_numpy()!
    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flip the given numpy ndarray.
    Args:
        img (numpy ndarray): Image to be flipped.
    Returns:
        numpy ndarray:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    return cv2.flip(img, 0)


def resized_crop(img, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
    """Crop the given numpy ndarray and resize it to desired size.
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.
    Args:
        img (numpy ndarray): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be numpy image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation=interpolation)
    return img


def resize(img, size, interpolation=cv2.INTER_LINEAR):
    r"""Resize the input numpy ndarray to the given size.
    Args:
        img (numpy ndarray): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))
    h, w, c = img.shape[:3]

    if isinstance(size, int):
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        ow, oh = size[1], size[0]
    
    output = cv2.resize(img, dsize=(ow, oh), interpolation=interpolation)

    if c == 1: output = output[..., None]

    return output


def rotate(img, angle, center=None):
    """Rotate the image by angle.
    Args:
        img (numpy ndarray): numpy ndarray to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    rows, cols = img.shape[0:2]
    if center is None:
        center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1) 
    if img.shape[2] == 1:
        return cv2.warpAffine(img, M, (cols, rows))[:, :, np.newaxis]
    else:
        return cv2.warpAffine(img, M, (cols, rows))


def hu2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new


def load_dcm(sorted_dcm_list):
    """
    Return img array and [z,y,x]-ordered origin and spacing
    """
    itkimage = sitk.ReadImage(sorted_dcm_list)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing
