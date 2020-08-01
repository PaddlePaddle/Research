GraphSum
===
Code for ACL2020 paper: [Leveraging Graph to Improve Abstractive Multi-Document Summarization](https://arxiv.org/pdf/2005.10043.pdf)

Introduction
---
GraphSum is a multi-document abstractive summarization model which integrates explicit graph representations of documents such as lexical similarity and discourse graph into transformer-based end-to-end neural generation model. 
The model can leverage graph to better encode long input from multiple documents and model the intra- and cross-document relations more effectively.
The model is proposed in ACL2020 paper [Leveraging Graph to Improve Abstractive Multi-Document Summarization](https://arxiv.org/pdf/2005.10043.pdf). 

Dependencies
---
* Python3.6.9  
* paddlepaddle-gpu==1.6.3.post107  
* pyrouge==0.1.3  
* sentencepiece==0.1.91 

Data Preparation
---
We have already processed WikiSum and MultiNews dataset. 
The processed WikiSum dataset can be downloaded from [https://graphsum.bj.bcebos.com/data/WikiSum_data_tfidf_40_paddle.tar.gz](https://graphsum.bj.bcebos.com/data/WikiSum_data_tfidf_40_paddle.tar.gz)

The processed MultiNews dataset can be downloaded from [https://graphsum.bj.bcebos.com/data/MultiNews_data_tfidf_30_paddle.tar.gz](https://graphsum.bj.bcebos.com/data/MultiNews_data_tfidf_30_paddle.tar.gz) (unzipped, and set $TASK_DATA_PATH in ./model_config/graphsum_model_conf_local).  

The sentencepiece vocab file can be downloaded from [https://graphsum.bj.bcebos.com/vocab/spm9998_3.model](https://graphsum.bj.bcebos.com/vocab/spm9998_3.model) (set $VOCAB_PATH in ./model_config/graphsum_model_conf_local).

To process the dataset and build graph by yourself, you can get the raw version of the WikiSum dataset from [here](https://github.com/tensorflow/tensor2tensor/tree/5acf4a44cc2cbe91cd788734075376af0f8dd3f4/tensor2tensor/data_generators/wikisum)
and the ranked version of the dataset from [here](https://github.com/nlpyang/hiersumm). The raw version of the MultiNews dataset can be obtained from [this link](https://github.com/Alex-Fabbri/Multi-News)

Our data preprocess code (contains the code to build similarity graph and topic graph) has also been open [link](https://github.com/weili-ict/Research/tree/master/NLP/ACL2020-GraphSum/src/data_preprocess/graphsum).


Model Configuration
---
Global Configuration for GraphSum model, such as hidden size, num of layers, head size etc.: 
```
./model_config/graphsum_config.json
```

Training and Testing Configurations, such as batch_size, num of training epoches, beam_size etc.
```
./model_config/graphsum_model_conf_local
```


Train
---
We use 8 Tesla-V100-32G GPUs to train our model. Directly runing the scripts as following to train the model:

Training on the WikiSum dataset:
```
./scripts/run_graphsum_local_wikisum.sh
```

Training on the MultiNews dataset:
```
./scripts/run_graphsum_local_multinews.sh
```


Test
---
After completing the training process, several checkpoints are stored, e.g. ./models/graphsum_wikisum/step_300000. 
You can run the following command to get the results on test set (only one GPU is required for testing):

Testing on the WikiSum dataset:
```
./scripts/predict_graphsum_local_wikisum.sh
```

Testing on the MultiNews dataset:
```
./scripts/predict_graphsum_local_multinews.sh
```

Results
---

|   dataset   | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ----------- | ------- | ------- | ------- |
|   MultiNews |  45.71  |  17.12  |  41.99  |
|   WikiSum   |  43.23  |  27.25  |  40.99  |

We re-trained the model by tuning some parameters, and got the above results which were a little better than the results reported in the paper.  
\*\*Note that, the above ROUGEL-L results are **sentence-level** ROUGE-L scores which are the most widely used in summarization research.


Generated Summaries
---
The generated summaries for GraphSum on the WikiSum dataset can be downloaded [https://graphsum.bj.bcebos.com/generated_summaries/graphsum_wikisum_results.tar.gz](https://graphsum.bj.bcebos.com/generated_summaries/graphsum_wikisum_results.tar.gz)  
The generated summaries for GraphSum on the MultiNews dataset can be downloaded [https://graphsum.bj.bcebos.com/generated_summaries/graphsum_multinews_results.tar.gz](https://graphsum.bj.bcebos.com/generated_summaries/graphsum_multinews_results.tar.gz)


Trained Models
---
The trained GraphSum model on the WikiSum dataset can be downloaded from [graphsum_wikisum_trained_model](https://graphsum.bj.bcebos.com/trained_models/graphsum_wikisum/step_308000.tar.gz)  
The trained GraphSum model on the MultiNews dataset can be downloaded from [graphsum_multinews_trained_model](https://graphsum.bj.bcebos.com/trained_models/graphsum_multinews/step_42976.tar.gz)  
To predict with our trained models, firstly unzip them and then set parameter $init_checkpoint in ./scripts/predict_graphsum_local_wikisum.sh and ./scripts/predict_graphsum_local_multinews.sh to the path of trained models, respectively.

Citation
---
If you find our code useful in your work, please cite the following paper:
>@article{li2020leveraging,  
  >title={Leveraging Graph to Improve Abstractive Multi-Document Summarization},  
  >author={Li, Wei and Xiao, Xinyan and Liu, Jiachen and Wu, Hua and Wang, Haifeng and Du, Junping},  
  >journal={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},  
  >year={2020},  
  >publisher={Association for Computational Linguistics}
>}  

