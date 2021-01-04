# SemSegPaddle-Dygraph: A Paddle-based Framework for Deep Learning in Semantic Segmentation

This repo is the dynamic graph version of [SemSegPaddle](https://github.com/PaddlePaddle/Research/tree/master/CV/SemSegPaddle).

## Update

- [**2020/12/22**] We release ***PSPNet-ResNet101*** and ***GloRe-ResNet101*** models on Pascal Context and Cityscapes datasets ( ***DynamicGraph***).
- [**2020/01/08**] We release ***PSPNet-ResNet101*** and ***GloRe-ResNet101*** models on Pascal Context and Cityscapes datasets (StaticGraph).

  
## Highlights

Synchronized Batch Normlization is important for segmenation.
  - The implementation is easy to use as it is pure-python, no any C++ extra extension libs.
   
  - Paddle provides sync_batch_norm.
   
   
## Support models

We split our models into backbone and decoder network, where backbone network are transfered from classification networks.

Backbone:
  - ResNet
  - ResNeXt
  - HRNet
  - EfficientNet
  
Decoder:
  - PSPNet: [Pyramid Scene Parsing Network](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)
  - DeepLabv3: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
  - GloRe: [Graph-Based Global Reasoning Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Graph-Based_Global_Reasoning_Networks_CVPR_2019_paper.pdf)
  - GINet: [GINet: Graph Interaction Netowrk for Scene Parsing](https://arxiv.org/pdf/2009.06160.pdf)
  



## Environment

This repo is developed under the following configurations:

 - Hardware: 1/2/4 GPU for training, 1 GPU for testing
 - Software: Centos 6.10, ***CUDA>=9.2 Python>=3.6, Paddle>=2.0***


## Quick start: training and testing models

### 1. Preparing data


Cityscapes

train_set:2975，val_set:500, test_set:1525，resolution:1024*2048.

Datasets Source：AIstudio  [Download](https://aistudio.baidu.com/aistudio/datasetDetail/11503), unzip cityscapes.zip to dir named 'dataset', unzip train.zip to cityscapes/leftImg8bit，directory structure is as follows：

```text
dataset
  ├── cityscapes               # Cityscapes dir
         ├── gtFine            # label
         ├── leftImg8bit       # images
         ├── trainLabels.txt   # training split
         ├── valLabels.txt     # validation split
              ...               ...
```

Pascal Context

train_set:4998, val_set:5105, test_set:9637.

Datasets Source : Standford  [Download](https://www.cs.stanford.edu/~roozbeh/pascal-context/)

Directory structure is as follows:

```text
pascal_context
  ├── GroundTruth_trainval_png
  ├── ImageSets
  ├── JPEGImages
  ├── pascal_context_train.txt
  ├── pascal_context_val.txt
      ...   
```
 
### 2. Download pretrained weights

Downlaod pretrained [ResNet101](https://pan.baidu.com/s/1rX5FEEjkPB5EObyd6LM-VQ) (提取码: wmpy), and put it into the directory: ***./pretrained_model***
  


### 3. Training

select confiure file for training according to the DECODER\_NAME, BACKBONE\_NAME and DATASET\_NAME. and change the data loader path and pretrained model path according to your setting.
```shell
# single_card:
CUDA_VISIBLE_DEVICES=0 python train.py  --use_gpu \
                                  --cfg ./configs/pspnet_res101_cityscapes.yaml 
# multi_card:
python -m paddle.distributed.launch --selected_gpus=0,1 train.py \
                                  --use_gpu --cfg configs/pspnet_resnet101_cityscapes.yaml
```

### 4. Testing 
select confiure file for testing according to the DECODER\_NAME, BACKBONE\_NAME and DATASET\_NAME.

Single-scale testing:
```shell
CUDA_VISIBLE_DEVICES=0 python  eval.py --use_gpu \
                                       --cfg ./configs/eval_pspnet_res101_cityscapes.yaml 
```


## Contact
If you have any questions regarding the repo, please create an issue.

