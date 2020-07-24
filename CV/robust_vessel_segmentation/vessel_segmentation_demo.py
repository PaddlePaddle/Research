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
"""
Demo for robust vessel segmentation.
Note: The vessel segmentation model used here is trained with two novel data augmentation methods, 
namely, channel-wise random gamma correction and channel-wise random vessel augmentation.
"""
import cv2
import numpy as np
import paddle.fluid as fluid


def resize_img(img, max_size=640):
    """
    Reisze the input image. 
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    max_axis = max(img_height, img_width)
    if max_axis > max_size:
        if img_height > img_width:
            img = cv2.resize(img, (int(img.shape[1] / img.shape[0] * max_size), max_size))
        else:
            img = cv2.resize(img, (max_size, int(img.shape[0] / img.shape[1] * max_size)))

    return img


def padding_img(img, size=640, padding_value=[127.5, 127.5, 127.5]):
    """
    Padding the input image.
    """
    img_height = img.shape[0]
    img_width = img.shape[1]
    pad_height = max(size - img_height, 0)
    pad_width = max(size - img_width, 0)
    if (pad_height > 0 or pad_width > 0):
        img = cv2.copyMakeBorder(
            img,
            0,
            pad_height,
            0,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=padding_value)
    else:
        raise Exception(
                "padding size({},{}) must large than img size({},{})."
                .format(size, size, img_width, img_height))
    
    return img


class VesselSegmentation(object):
    """
    Segment vessel mask from ori fundus photography imgage.
    Change states: self.vessel_segmentation_mask, i indicates vessel and 0 otherwise.
                   self.vessel_masked_img, the mask imge
    SWE: sunxu02
    """

    def __init__(self, 
                 dirname, 
                 params_filename, 
                 img_mean=[127.5, 127.5, 127.5], 
                 img_std=[127.5, 127.5, 127.5], 
                 use_cuda=True, 
                 final_size=640, 
                 vessel_thresh=0.5,
                 vessel_color=[255, 255, 255]):
        # initialize parameters to load model
        self.use_cuda = use_cuda
        self.dirname = dirname
        self.params_filename = params_filename
        self._load_model()
        # initialize parameters for image preprocessing
        self.img_mean = img_mean
        self.img_std = img_std
        self.final_size = final_size
        # initialize parameters for image postprocessing
        self.vessel_thresh = vessel_thresh
        self.vessel_color = vessel_color

    def _load_model(self):
        paddle_place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
        self.paddle_exe = fluid.Executor(paddle_place)

        self.vs_scope = fluid.core.Scope()
        with fluid.scope_guard(self.vs_scope):
            [self.vs_net_paddle, self. vs_feed_names, self.vs_targets] = fluid.io.load_inference_model(
                dirname=self.dirname,
                executor=self.paddle_exe,
                params_filename=self.params_filename)

    def _preprocess(self):
        # format the image shape
        self.ori_shape = ori_img.shape
        img = resize_img(self.ori_img.copy(), self.final_size)
        self.valid_shape = img.shape
        img = padding_img(img, size=self.final_size, padding_value=self.img_mean)
        # normalize
        img = (img - self.img_mean) / self.img_std
        
        self.model_input = np.transpose(img, (2, 0, 1)).astype(np.float32)

    def _infer(self):
        model_input = np.expand_dims(self.model_input, 0)
        with fluid.scope_guard(self.vs_scope):
            output = self.paddle_exe.run(self.vs_net_paddle,
                                         feed={self.vs_feed_names[0]: model_input},
                                         fetch_list=self.vs_targets)
        self.vessel_mask = np.array(output[0][0][1])

    def _postprocess(self):
        # extractt the vessel mask and convert it to the original size
        vessel_mask = self.vessel_mask[:self.valid_shape[0], :self.valid_shape[1]]
        vessel_mask = cv2.resize(vessel_mask, (self.ori_shape[1], self.ori_shape[0]))
        # get the binary mask
        vessel_mask = np.uint8(vessel_mask > self.vessel_thresh)
        # masked the original image
        masked_img = self.ori_img.copy()
        masked_img[vessel_mask==1] = self.vessel_color
        # save the results
        self.vessel_segmentation_mask = vessel_mask
        self.vessel_masked_img = masked_img

    def __call__(self, img):
        self.ori_img = img.copy()
        self._preprocess()
        self._infer()
        self._postprocess()
        return self.vessel_segmentation_mask, self.vessel_masked_img


if __name__ == '__main__':

    ori_img_file = 'demo.jpg'
    dirname = 'pdl_assets'
    params_file_name = 'pdl_assets/__params__'

    vs_model = VesselSegmentation(dirname, params_file_name)
    ori_img = cv2.imread(ori_img_file)
    
    _, vessel_masked_img = vs_model(ori_img)
    cv2.imwrite('masked_img.png', vessel_masked_img)
