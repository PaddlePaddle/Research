# AICity Challenge 2020 Track2 vehicle re-identification(re-id).

This branch contains paddlepaddle training codes for aicity 2020 Track2 vehicle re-identification. More detail can be found in [aicity](https://www.aicitychallenge.org/)

Two architectures of feature net are supported:

* single branch(traditional reid model)
* [multi branch](https://github.com/douzi0248/Re-ID)

Besides, multi-task learning (reid, color, type) is also supported

Imagenet pretrained models can be found in [PaddleCV](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification) 

## 1. Environment installation
* CUDA 9
* cuDNN >=7.3
* paddlepaddle-gpu == 1.6.3

## 2. Process aicity 2020 re-id data
* tools are put in process_aicity_data directory, please see README in this directory

## 3. Training
1. modify configurations in config.py or arguments in train.sh
2. `sh train.sh`


## 4 Testing 
1. modify configurations in config.py or arguments in test.sh
2. `sh test.sh`