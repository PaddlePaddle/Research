# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import random
import functools
import math
import cv2

import numpy as np
import pickle as pk
import pandas as pd
import paddle
import paddle.fluid as fluid

from clfnet import create_model
from parser import args_parser
import opencv_transforms as transforms

def create_evaluate_graph(use_cuda, args):
    """
    Create Paddle Compute Graph for evaluate
    """
    assert os.path.exists(args.resumed_checkpoint_path), "Pretrained model path {} not exist!".format(
        args.resumed_checkpoint_path)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    startup_prog = fluid.Program()
    val_prog = fluid.Program()

    with fluid.program_guard(val_prog, startup_prog):
        # Use fluid.unique_name.guard() to share parameters with train network
        with fluid.unique_name.guard():
            clf = create_model(data_shape=[1024, 1024, 3], 
                                 loss_type=args.loss_type, 
                                 main_arch=args.arch,
                                 name='infer')
            # [loss, accuracy, predict, hacked_img_id, cam_heatmap], reader
            val_tensor_collector, val_reader = clf.net(class_dim=int(args.infer_classdim), CAM=True)

    for tensor in val_tensor_collector:
        tensor.persistable = True

    val_prog = val_prog.clone(for_test=True)

    # Executor
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    def func(var):
        return os.path.exists(os.path.join(args.resumed_checkpoint_path, var.name))
    fluid.io.load_vars(
        exe, args.resumed_checkpoint_path, main_program=val_prog, predicate=func)

    return exe, (val_prog, val_tensor_collector, val_reader)

def inference(exe, val_prog, file_paths, fetch_list):
    """
    load images and feed into Compute Graph, accumulate running results.
    """
    prediction = []
    cam_heatmaps = []
    

    trans = transforms.Compose([
        transforms.ResizeShort(size=1024),
        transforms.CenterCrop(size=1024),
        transforms.LocalMedianSubtract(size=1024, radius=90)
    ])

    for img_id, file_path in enumerate(file_paths):
        img = cv2.imread(file_path).astype("uint8")[:, :, ::-1]
        img = trans(img)
        img = np.expand_dims(img, 0)

        val_out = exe.run(
            program=val_prog,
            feed={"infer_input": img, 
                  "infer_label": np.array(0).reshape(1, 1), 
                  "infer_img_id": np.array(img_id).reshape(1, 1)},
            fetch_list=fetch_list, #[val_loss, val_acc, val_output, val_img_id, cam_heatmaps]
            use_program_cache=True)

        prediction.append(val_out[2][0])
        cam_heatmaps.append(val_out[4][0])

    return prediction, cam_heatmaps

def render_cam(img_paths, cams):
    """
    Render Class Activation Map to png.
    """
    trans = transforms.Compose([
        transforms.ResizeShort(size=1024),
        transforms.CenterCrop(size=1024),
    ])

    for file, cam in zip(img_paths, cams):
        img = cv2.imread(file).astype("uint8")
        img = trans(img)

        cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
        cam = cv2.resize(cam, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        cam = cv2.cvtColor(cam, cv2.COLOR_GRAY2RGB)

        cam_img = img * (cam / 255)
        cv2.imwrite("./cam_heatmaps/{}.png".format(file.split("/")[-1].split(".")[0]),
                    np.concatenate([img, cam, cam_img], 1))

def main(args):
    """Main"""
    fluid.install_check.run_check()
    use_cuda = True

    exe, val_prog_tensors = create_evaluate_graph(use_cuda, args)

    val_prog, val_tensor_collector, val_reader = val_prog_tensors

    file_paths = list(map(str.strip, open(args.infer_file, "r").readlines()))
    prediction, cam_heatmaps = inference(exe, val_prog, file_paths, 
                           fetch_list=val_tensor_collector)

    if not os.path.exists("./cam_heatmaps"):
        os.mkdir("./cam_heatmaps")

    render_cam(file_paths, cam_heatmaps)

if __name__ == '__main__':
    args = args_parser().parse_args()
    print(args)

    # Infer
    main(args)