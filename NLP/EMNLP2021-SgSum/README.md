SgSum
===
Code for EMNLP2021 paper: [SgSum:Transforming Multi-document Summarization into Sub-graph Selection](https://arxiv.org/abs/2110.12645)

Introduction
---
SgSum is a MDS framework formulates the MDS task as a sub-graph selection problem, in which source documents are regarded as a relation graph of sentences (e.g., similarity graph or discourse graph) and the candidate summaries are its sub-graphs. Instead of selecting salient sentences, SgSum selects a salient sub-graph from the relation graph as the summary. Comparing with traditional methods, our method has two main advantages: 

(1) The relations between sentences are captured by modeling both the graph structure of the whole document set and the candidate sub-graphs

(2) Directly outputs an integrate summary in the form of sub-graph which is more informative and coherent

Dependencies
---
* Python3.7.6
* paddlepaddle-gpu==1.8.3.post107  
* pyrouge==0.1.3  
* sentencepiece==0.1.91
* nltk==3.5
* gensim==3.8.3

Data Preparation
---
We have already processed MultiNews dataset. 
The processed MultiNews dataset can be downloaded from [http://sgsum.bj.bcebos.com/public_files/multinews_dataset.tar.gz](https://sgsum.bj.bcebos.com/public_files/multinews_dataset.tar.gz?authorization=bce-auth-v1/744995617a4d4a249f62b9286323c96e/2021-10-14T12%3A12%3A10Z/-1/host/3888c457800ca86b73a0c87261c06b81d0f563cfeea3a1a4239659630add1e06) (unzipped, and set $TASK_DATA_PATH in ./model_config/roberta_graphsum_model_conf).  

To process the dataset and build graph by yourself, you can get the raw version of the MultiNews dataset can be obtained from [this link](https://github.com/Alex-Fabbri/Multi-News)

Our data preprocess code (contains the code to build similarity graph and topic graph) has also been open in data_preprocess dir.


Model Configuration
---
Global Configuration for GraphSum model, such as hidden size, num of layers, head size etc.: 
```
./model_config/graphsum_config.json
```

Training and Testing Configurations, such as batch_size, num of training epoches, batch_size etc.
```
./model_config/roberta_graphsum_model_conf
```

RoBERTa Pretrained Model
---
We use base version of RoBERTa model to initialize our models in all experiments. The RoBERTa model can be downloaded from [RoBERTa base version](https://sgsum.bj.bcebos.com/public_files/roberta_config.tar.gz?authorization=bce-auth-v1/744995617a4d4a249f62b9286323c96e/2021-10-14T12%3A25%3A02Z/-1/host/ad9581f065a2182970abdd27f5f080e37efb3c31bb7ba716497e3082d7dc1f38)


Train
---
We use 4 Tesla-V100-32G GPUs to train our model. Directly runing the scripts as following to train the model:

Training on the MultiNews dataset:
```
./scripts/train.sh
```


Test
---
After completing the training process, several checkpoints are stored, e.g. ./model_checkpoints/step_21000. 
You can run the following command to get the results on test set (only one GPU is required for testing):

Testing on the MultiNews dataset:
```
./scripts/predict.sh
```

Results
---

|   dataset   | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ----------- | ------- | ------- | ------- |
|   MultiNews |  47.36  |  18.61  |  43.13  |
|   DUC2004   |  38.66  |   9.73  |  34.02  |

\*\*Note that, the above ROUGEL-L results are **sentence-level** ROUGE-L scores which are the most widely used in summarization research.


Generated Summaries
---
The generated summaries for SgSum on the MultiNews dataset can be downloaded [https://sgsum.bj.bcebos.com/public_files/multinews_outputs.tar.gz](https://sgsum.bj.bcebos.com/public_files/multinews_outputs.tar.gz?authorization=bce-auth-v1/744995617a4d4a249f62b9286323c96e/2021-10-14T12%3A12%3A44Z/-1/host/c95a0e669bdf2fd1ade87cf7f5a4fdb63e570abdccac0f8708ac41ef553274fb)

The generated summaries for SgSum on the DUC2004 dataset can be downloaded [https://sgsum.bj.bcebos.com/public_files/duc2004_outputs.tar.gz](https://sgsum.bj.bcebos.com/public_files/duc2004_outputs.tar.gz?authorization=bce-auth-v1/744995617a4d4a249f62b9286323c96e/2021-10-14T12%3A13%3A09Z/-1/host/e43827378379c59405a64ed1e2c158b80a7bd772b64af41752626d5e65edece0)  

Trained Models
---
The MultiNews model can be downloaded from [multinews_trained_model](https://sgsum.bj.bcebos.com/public_files/multinews.tar.gz?authorization=bce-auth-v1/744995617a4d4a249f62b9286323c96e/2021-10-14T12%3A14%3A10Z/-1/host/dc772fa1dc66a44ab9a1d78d5cc2a75ba00aeaf27dbdf32eac4459642e9dcd13)  
The MultiNews-Extra model can be downloaded from [multinews_extra_trained_model](https://sgsum.bj.bcebos.com/public_files/multinews_extra.tar.gz?authorization=bce-auth-v1/744995617a4d4a249f62b9286323c96e/2021-10-14T12%3A14%3A40Z/-1/host/66b92ee834132b3ff7de460b5b6b4592aaacba503899f354e6bcf9c5b6518a10)  
To predict with our trained models, firstly unzip them and then set parameter $init_checkpoint in ./scripts/predict.sh to the path of trained models.

Citation
---
If you find our code useful in your work, please cite the following paper:
>@article{chen2021sgsum,  
  >title={SgSum: Transforming Multi-document Summarization into Sub-graph Selection},  
  >author={Moye Chen and Wei Li and Jiachen Liu and Xinyan Xiao and Hua Wu and Haifeng Wang},  
  >year={2021},  
  >eprint={2110.12645},
  >archivePrefix={arXiv},
  >primaryClass={cs.CL}
>}  