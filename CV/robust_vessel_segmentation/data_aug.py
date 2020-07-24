#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import numpy as np


def find_largest_connected_component_area(mask, connectivity=8):
    """
    Find the largest connected component area froma mask image.
    ----
    Args:
        mask: 2D array, 0 for background and 1 for foreground.
        connectivity: int, scalar, chose 4-connectivity or 8.
    Returns:
        output: int, 2D array, the largest connected component area.
    """
    labels, stats = cv2.connectedComponentsWithStats(mask, connectivity)[1:3]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
    output = np.uint8(labels == largest_label)
    
    return output


def get_fov(img, size=512, threshold=0.05):
    """
    Get the field of view from an fundus image.
    ----
    Args:
        img: uint8, 2D array, sugested be the red channel of a fundus image.
        size: int, scalar, the image size to be process in the procedure, 
              set it samller than the original size to make it faster.
        threshold: float (0-1), scalar, threshold to determine the foreground mask.
    Returns:
        mask: int (0 or 1), 2D array, the final mask.
    """
    img_shape = img.shape
    if img_shape[0] > size:
        img = cv2.resize(img, (size, size))
    mask = np.uint8((img / 255.0) > threshold)
    if np.sum(mask) == 0:
        print('Warning: None Foreground Detected')
    else:
        mask = find_largest_connected_component_area(mask)
    if img_shape[0] > size:
        mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
    mask = np.uint8(mask)

    return mask


def rotate_image(image, angle):
    """
    Rotate an image.
    ----
    Args:
        image: uint8, 2D numpy array range from 0 to 255, the input image
        angle: float, scalar, the rotation angle 
    Returns:
        result: unit8, 2D array, the rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1])
    
    return result


def morphological_transform(image, inverse=True, max_length=21, degree_num=8):
    """
    Apply Morphological transform on an image.
    ----
    Args:
        image: uint8, 2D numpy array range from 0 to 255, the input image
        inverse: bool, scalar, determine if to inverse the image intensity
        degree_num: int, scalar, number of degrees to apply to morphological transform
    Returns:
        tophat_sum: float, 2D array, the transformed image.
    """
    
    line = np.zeros((max_length, max_length)).astype(np.uint8)
    line[max_length // 2, :] = 1

    tophat_sum = np.zeros(image.shape).astype(np.uint16)
    for degree in np.arange(0, 181, 180 / degree_num):
        line_rotate = (rotate_image(line, degree))
        if inverse:
            tophat = cv2.morphologyEx(255 - image.copy(), cv2.MORPH_TOPHAT, line_rotate)
        else:
            tophat = cv2.morphologyEx(image.copy(), cv2.MORPH_TOPHAT, line_rotate)
        tophat_sum = tophat_sum + tophat
    
    tophat_sum = cv2.normalize(tophat_sum, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    tophat_sum = np.uint8(tophat_sum)
        
    return tophat_sum


def apply_vessel_aug(image, order='RGB'):
    """
    Apply vessel augmentation on an fundus image.
    ----
    Args:
        image: uint8, 3D array in RGB order, the original image.
    Returns:
        aug_image: uint8, 3D array in RGB order, the transformed image.
    """
    if order == 'RGB':
        red_data = image[:, :, 0]
    elif order == 'BGR':
        red_data = image[:, :, 2]
    else:
        assert False, print('Only support RBG or BGR image format.')
    
    green_data = image[:, :, 1]
    fundus_mask = get_fov(red_data)
    vessel_map = morphological_transform(green_data)
    vessel_map = vessel_map * fundus_mask
    vessel_map = cv2.GaussianBlur(vessel_map, ksize=(3, 3), sigmaX=0, sigmaY=0)
    # vessel_map = cv2.GaussianBlur(vessel_map, ksize=(5, 5), sigmaX=0, sigmaY=0)
    vessel_map[vessel_map > 204] = 204
    vessel_map = np.repeat(vessel_map[:, :, np.newaxis], 3, axis=2)
    random_decay = np.random.uniform(0, 1, 3)
    for i in range(3):
        vessel_map[:, :, i] = vessel_map[:, :, i] * random_decay[i]

    amp = np.random.uniform(0, 1, 1)
    aug_image = image.astype(float) * (1 - vessel_map / 255) + vessel_map * amp
    aug_image[aug_image > 255] = 255

    return np.uint8(aug_image)
    

def random_vessel_augmentation(image, prob=0.5, order='RGB'):
    """
    Channel-wise random vessel augmentation.
    """
    rand_num = np.random.uniform(0, 1)
    if rand_num > prob:
        image = apply_vessel_aug(image, order)
        
    return image


def random_gamma_correction(image, gamma=3, order='RGB'):
    """
    Channel-wise random gamma correction in the RGB color space.
    ----
    Args:
        image: uint8, 3D array in RGB order, the original image.
        gamma: maximum value of gamma for gamma corretion.
    Returns:
        image: uint8, 3D array in RGB order, the transformed image.
    """
    rds1 = np.random.uniform(1, gamma, 3)
    rds2 = np.random.uniform(0, 1, 3) < 0.5

    for i in range(3):
        if rds2[i]:
            rds1[i] = 1 / rds1[i]
            
    trans_image = np.power(image / 255.0, rds1)
    trans_image = trans_image * 255

    if order == 'RGB':
        red_data = image[:, :, 0]
    elif order == 'BGR':
        red_data = image[:, :, 2]
    else:
        assert False, print('Only support RBG or BGR image format.')
    
    fundus_mask = get_fov(red_data)
    trans_image = trans_image * fundus_mask[:, :, None]
    
    return np.uint8(trans_image)