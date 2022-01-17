#!/usr/bin/env python3
# coding=utf-8
"""
Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
File: infer.py
func: 测试代码,高版本训练低版本预测
Author: yuwei09(yuwei09@baidu.com)
Date: 2021/08/06
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import PIL
from PIL import Image

from mmflib.arch.arch import MMFModel

class InferenceModel(nn.Module):
    """Inference model
    Args:
        config: 配置参数
        mode: ['h', 'l'] 'h' 高版本torch预测， 'l' 低版本torch预测
    """
    def __init__(self, config, mode='h'):
        super(InferenceModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.model = self.build_model()
        print(self.model)
        if mode == 'h':
            self.load_pth_h()
        else:
            self.load_pth_l()
        self.model.eval()

    def forward(self, img_path, word):
        """forward"""
        image = self.read_img(img_path)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).to(self.device, dtype=torch.float)
        with torch.no_grad():
            ret_model = self.model(image, word)
            embeddings = ret_model["features"]
            embeddings_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            embeddings = embeddings.div(embeddings_norm.expand_as(embeddings))
            embeddings_norm = embeddings_norm.data.cpu()[0].numpy()   
            embeddings=embeddings.data.cpu()[0].numpy() 

        return embeddings

    def build_model(self):
        """构建模型
        Returns:
           arch: 返回模型结构 
        """
        config = copy.deepcopy(self.config["Arch"])
        arc_name = config.pop("name")
        arch = eval(arc_name)(config, mode="infer")

        return arch

    def unzip_pth(self):
        """导入高版本pth转存为低版本pth
        """
        save_path = self.config["Global"]["pretrained_model"][:-4] + '_infer.pth'
        torch.save(self.model, save_path, _use_new_zipfile_serialization=False)
        print("Model saved as %s" % save_path)

    def load_pth_h(self):
        """导入高版本pth"""
        pth_path = self.config["Global"]["pretrained_model"]
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
        ###仅加载backbone参数
        model_dict = self.model.state_dict()
        state_dict = {k:v for k, v in checkpoint['model_state'].items() if 'head.weight' not in k}
        model_dict.update(state_dict)
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        print("Model restored from %s" % pth_path)
    
    def load_pth_l(self):
        "导入低版本pth"
        pth_path = self.config["Global"]["pretrained_model"]
        checkpoint = torch.load(pth_path)
        self.model.load_state_dict(checkpoint.state_dict())
        self.model.to(self.device)
        print("Model restored from %s" % pth_path)

    def read_img(self, path):
        """读取图像
        Args:
            path: 图像路径
        """
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224), resample=PIL.Image.BICUBIC)
        img = np.array(img)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))

        return img