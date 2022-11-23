# SMILE: Self-Distilled MIxup for Efficient Transfer LEarning
## Introduction

This is the [PaddlePaddle](https://www.paddlepaddle.org.cn/) implementation of the SMILE (Spotlight on [INTERPOLATE@NeurIPS 2022](https://sites.google.com/view/interpolation-workshop?pli=1)) model for image classification.

In this work, we propose SMILE— Self-Distilled Mixup for EffIcient Transfer LEarning.
With mixed images as inputs, SMILE regularizes the outputs of CNN feature extractors to learn
from the mixed feature vectors of inputs (sample-to-feature mixup), in addition to the mixed labels.
Specifically, SMILE incorporates a mean teacher, inherited from the pre-trained model, to provide
the feature vectors of input samples in a self-distilling fashion, and mixes up the feature vectors
accordingly via a novel triplet regularizer. The triple regularizer balances the mixup effects in both
feature and label spaces while bounding the linearity in-between samples for pre-training tasks.



## Requirements
The code has been tested running under the following environments:

* python >= 3.7
* numpy >= 1.21
* paddlepaddle >= 2.2 (with suitable CUDA and cuDNN version)
* visualdl
 


## Model Training

### step1. Download dataset files
We conduct experiments on three popular object recognition datasets: CUB-200-2011, Stanford Cars and
FGVC-Aircraft. You can download it from the official link below.
 - [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/) 
 - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
 - [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

Please organize your dataset in the following format. 
```
dataset
├── train
│   ├── class_001
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   ├── class_002
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   └── ...
└── test
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```

### step2. Finetune

You can use the following command to finetune the target data using the SMILE algorithm. Log files and ckpts during training are saved in the ./output. Only the model with the highest accuracy on the validation set is saved during finetuning.
```
python finetune.py --name {name of your experiment} --train_dir {path of train dir} --eval_dir {path of eval dir} --model_arch resnet50 --gpu {gpu id} --regularizer smile
```

### step3. Test

You can also load the finetuning ckpts with the following command and test it on the test set.
```
python test.py --test_dir {path of test dir} --model_arch resnet50 --gpu {gpu id} --ckpts {path of finetuning ckpts}
```

## Results

|Dataset/Method | L2 | SMILE |
|---|---|---|
|CUB-200-2011 | 80.79 | 82.38 |
|Stanford-Cars| 90.72 | 91.74 |
|FGVC-Aircraft| 86.93 | 89.00 |



## Citation
If you use any source code included in this project in your work, please cite the following paper:

```
@article{Li2021SMILESM,
  title={SMILE: Self-Distilled MIxup for Efficient Transfer LEarning},
  author={Xingjian Li and Haoyi Xiong and Chengzhong Xu and Dejing Dou},
  journal={ArXiv},
  year={2021},
  volume={abs/2103.13941}
}
```

## Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.