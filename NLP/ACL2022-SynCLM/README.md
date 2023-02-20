SynCLM
====
Code for Findings of ACL 2022 long paper: [Syntax-guided Contrastive Learning for Pre-trained Language Model](https://aclanthology.org/2022.findings-acl.191/)




Abstract
---
Syntactic information has been proved to be useful for transformer-based pre-trained language models. Previous studies often rely on additional syntax-guided attention components to enhance the transformer, which require more parameters and additional syntactic parsing in downstream tasks. This increase in complexity severely limits the application of syntax-enhanced language model in a wide range of scenarios. In order to inject syntactic knowledge effectively and efficiently into pre-trained language models, we propose a novel syntax-guided contrastive learning method which does not change the transformer architecture. Based on constituency and dependency structures of syntax trees, we design phrase-guided and tree-guided contrastive objectives, and optimize them in the pre-training stage, so as to help the pre-trained language model to capture rich syntactic knowledge in its representations. Experimental results show that our contrastive method achieves consistent improvements in a variety of tasks, including grammatical error detection, entity tasks, structural probing and GLUE. Detailed analysis further verifies that the improvements come from the utilization of syntactic information, and the learned attention weights are more explainable in terms of linguistics.


![SynCLM](images/framework.png#pic_center)



Dependencies
---
python3.7.4\
cuda-10.1\
cudnn_v7.6\
nccl2.4.2\
java1.8
paddlepaddle-gpu2.1.2\
stanza1.2\
numpy1.20.2



Pre-trained Models
---
SynCLM is trained based on RoBERTa model, users can use the following command to download the paddle version of RoBERTa model:

```shell
cd /path/to/model_files
# download base model
sh ./download_roberta_base_en.sh
# or download large model
# sh ./download_roberta_large_en.sh
cd -
```
To obtain the syntactic structures of the text, we use [Stanza](https://github.com/stanfordnlp/stanza)  to preprocess a data which is English Wikipedia and BookCorpus. We provide input examples in the `/path/to/data/pretrain` directory.

After preparing the data, you can run the following command for training:
```shell
cd /path/to
# base model
sh ./script/roberta_base_en/run.sh
# or large model
# sh ./script/roberta_large_en/run.sh
```
After pre-training the model, users can use the following command to fine-tune it on downstream tasks：
```shell
# classification
python ./src/run_classifier.py
# regression
python ./src/run_regression.py
```


Citation
---
If you find our paper and code useful, please cite the following paper:
```
@inproceedings{zhang2022syntax,
  title={Syntax-guided Contrastive Learning for Pre-trained Language Model},
  author={Zhang, Shuai and Lijie, Wang and Xiao, Xinyan and Wu, Hua},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  pages={2430--2440},
  year={2022}
}
```

