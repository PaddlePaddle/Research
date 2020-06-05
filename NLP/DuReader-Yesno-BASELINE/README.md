# DuReader<sub>yesno</sub> Dataset
DuReader<sub>yesno</sub> dataset is based on the yes-no samples in [DuReader](https://arxiv.org/abs/1711.05073). For these samples, a traditional span extraction model may be able to predict the correct answer span in the paragraph, but whether the model is capable of understanding the "opinion" in the answer is unclear.  Thus the DuReader<sub>yesno</sub> dataset requires a model to predict the opinion (i.e. *Yes*, *No* or *Depends*) of the human annotated answers to a yes-no question. In the meanwhile, a few documents from which the human annotated answer comes are also provied. This dataset is a good complement to the span extraction datasets, and can be used to better evaluate the language understating capability of a MRC model.

An example from the dataset is shown below:

```
{
    "documents":[
        {
            "title":"香蕉能放冰箱吗 香蕉剥皮冷冻保存_健康贴士_保健_99健康网",
            "paragraphs":[
                "本文导读:............."
            ]
        }
    ],
    "yesno_answer":"No",
    "question":"香蕉能放冰箱吗",
    "answer":"香蕉不能放冰箱，香蕉如果放冰箱里，会更容易变坏，会发黑腐烂。",
    "id":293  
}
```
- id: Unique ID of this sample，type int；
- question: a yes-no question from user, type string；
- answer: human annotated answer segment，type string；
- yesno_answer：yes-no opinion of the answer，only three values are allowed, i.e. {“ Yes”, “No“, “Depends“}, type string, **case sensitive**；
- documents: retrieved documents from Baidu search engin that are related to the question, from which the answer is annotated.

# DuReader<sub>yesno</sub> Baseline System

In this repository, we release a baseline system for DuReader<sub>yesno</sub> dataset. The baseline system is based on [ERNIE 1.0](https://arxiv.org/abs/1904.09223), and is implemented with [PaddlePaddle](https://www.paddlepaddle.org.cn/) framework. To run the baseline system, please follow the instructions below.

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
Before run the baseline system, please download the DuReader<sub>yesno</sub> dataset and the pretrained model parameters (ERNIE 1.0 base):

```
sh download.sh
```
The dataset will be saved into `data/`, the pretrained and fine-tuned model parameters will be saved into `ernie/` and `baseline/`, respectively. Note that the fine-tuned model was fine-tuned from the training data based on the pretrained model (ERNIE 1.0). 

## Run Baseline

### Training
To fine-tune a model (on the demo dataset), please run the following command:

```shell script
sh run.sh --do_train true --do_val true --do_test true --train_set data/train.demo.json --dev_set data/dev.demo.json --test_set data/test.demo.json
```

This will start the training process on the demo dataset. At the end of training, model parameters and predictions will be saved into `output/`.

To train the model with user specified arguments (e.g. multi-GPUs, more epochs and different batch size), please run the following command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run.sh --epoch 10 --batch_size 8 --do_train true --do_val true --do_test true --train_set data/train.demo.json --dev_set data/dev.demo.json --test_set data/test.demo.json 
```
More arguments can be found in `src/finetune_args.py`.


### Prediction
To predict with fine-tuned parameters, (e.g. on the devlopment set), please run the following command:

```
sh run.sh --do_test true --test_set data/dev.json --init_checkpoint baseline/step_23561
```
The model parameters under `baseline/step_23561` (as we have provided) will be loaded for prediction. The predicted answers will be saved into `output/`.

### Evaluation
Accuracy is used as evaluation metrics. Here we provide a script `evaluate.py` for evaluation.


#### Evaluate on the dev set
To evluate on the development set, run

```shell script
python evaluation.py <reference-file> <predict-file>
```

where `<reference-file>` is the development set which contain the reference label and `<predict-file>` are the model prediction. After runing the evaluation script, you will get the evaluation results with the following format:

```
{"errorCode": "0", "errorMsg": "success", "data": [{"name": "acc", "value": 85.24}]}
```

#### Evaluate on the test set
Since there is no reference label in the test set, prediction results have to be submitted to the [official website]() for evaluation. For more details, please refer to the instruction on the website.

#### Baseline Performance
The performance of the baseline model that we provie is shown in the table below

| Model |  Acc - dev | Acc - test |
| ---- | ---- | ---- |
|ERNIE |85.24%|85.95%|


# Copyright and License
Copyright 2020 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
