# DuReader<sub>robust</sub> Dataset
Machine Reading Comprehension (MRC) is a crucial and challenging task in natural language processing (NLP). In recent years, with the development of deep learning and the increasing availability of large-scale datasets, MRC has achieved remarkable advancements. Although a number of MRC models obtains human parity performance on several datasets, we find that these models are still far from robust in the scenario of real-world applications. 

To comprehensively evaluate the robustness of MRC models, we create a Chinese dataset, namely DuReader<sub>robust</sub>. It is designed to challenge MRC models from the following aspects: (1) over-sensitivity, (2) over-stability and (3) generalization. Besides, DuReader<sub>robust</sub> has another advantage over previous datasets: questions and documents are from Baidu Search. It presents the robustness issues of MRC models when applying them to real-world applications.  

For more details about the dataset, please refer to this [paper](#).

# DuReader<sub>robust</sub> Baseline System
In this repository, we release a baseline system for DuReader<sub>robust</sub> dataset. The baseline system is based on [ERNIE 1.0](https://arxiv.org/abs/1904.09223), and is implemented with [PaddlePaddle](https://www.paddlepaddle.org.cn/) framework. To run the baseline system, please follow the instructions below.

## Environment Requirements
The baseline system has been tested on

 - CentOS 6.3
 - PaddlePaddle 1.6.1 
 - Python 2.7.13 & Python 3.7.3
 - Cuda 9.0
 - CuDnn 7.0
 - NCCL 2.2.13
 
To install PaddlePaddle, please see the [PaddlePaddle Homepage](http://paddlepaddle.org/) for more information.


## Download
Before run the baseline system, please download the DuReader<sub>robust</sub> dataset and the pretrained model parameters (ERNIE 1.0 base):

```
sh download.sh
```
The dataset will be saved into `data/`, the pretrained and fine-tuned model parameters will be saved into `pretrained_model/` and `finetuned_model/`, respectively. Note that the fine-tuned model was fine-tuned from the training data based on the pretrained model (ERNIE 1.0). 

The descriptions of the data structure can be found in `data/README.md`. 

## Run Baseline

### Training
To fine-tune a model (on the demo dataset), please run the following command:

```
sh train.sh --train_file data/demo/demo_train.json --predict_file data/demo/demo_dev.json 
```
This will start the training process on the demo dataset. At the end of training, model parameters and predictions will be saved into `output/`. 

To train the model with user specified arguments (e.g. multi-GPUs, more epochs and larger batch size), please run the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh train.sh --epoch 5 --batch_size 8 --train_file data/demo/demo_train.json --predict_file data/demo/demo_dev.json 
```
You can also directly modify the arguments in `train.sh`. More arguments can be found in `src/run_mrc.py`.

### Prediction
To predict with fine-tuned parameters, (e.g. on the devlopment set), please run the following command:

```
CKPT=<your_model_dir> sh predict.sh --predict_file data/dev.json
```
The model parameters under `your_model_dir` (e.g. `finetuned_model` as we have provided) will be loaded for prediction. The predicted answers will be saved into `output/`.

## Evaluation
F1-score and exact match (EM) are used as evaluation metrics. Here we provide a script `evaluate.py` for evaluation.

To evluate, run

```
python evaluate.py <reference_file> <prediction_file>
```
Where `<reference_file>` is the dataset file (e.g. `data/dev.json`), and `<prediction_file>` is the model prediction (e.g. `output/dev_predictions.json`) that should be a valid JSON file of (qid, answer) pairs, for example:

```
{
    "question_id_1" : "answer for question 1",
    ...
    "question_id_N": "answer for question N"
}
```

After runing the evaluation script, you will get the evaluation results with the following format:

```
{"F1": "80.842", "EM": "69.019", "TOTAL": 1417, "SKIP": 0}
```

## Baseline Performance
The performance of our baseline model (i.e. the fine-tuned model we provided above) are shown below:

| Dataset | F1 | EM |
| --- | --- | --- |
| basic dev | 80.84 | 69.02 |



## Copyright and License
Copyright 2020 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
