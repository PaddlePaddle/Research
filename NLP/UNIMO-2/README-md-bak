UNIMO
====
Code for the findings of ACL2022 long paper [UNIMO-2: End-to-End Unified Vision-Language Grounded Learning](https://arxiv.org/pdf/2203.09067.pdf)


Abstract
---

Vision-Language Pre-training (VLP) has achieved impressive performance on various cross-modal downstream tasks. 
However, most existing methods can only learn from aligned image-caption data and rely heavily on expensive regional 
features, which greatly limits their scalability and performance. In this paper, we propose an end-to-end unified-modal 
pre-training framework, namely UNIMO-2, for joint learning on both aligned image-caption data and unaligned image-only 
and text-only corpus. We build a unified Transformer model to jointly learn visual representations, textual 
representations and semantic alignment between images and texts. In particular, we propose to conduct grounded learning 
on both images and texts via a sharing grounded space, which helps bridge unaligned images and texts, and align the 
visual and textual semantic spaces on different types of corpora. The experiments show that our grounded learning 
method can improve textual and visual semantic alignment for improving performance on various cross-modal tasks. 
Moreover, benefiting from effective joint modeling of different types of corpora, our model also achieves impressive 
performance on single-modal visual and textual tasks. Our code and models are public at the UNIMO project page 
\url{https://unimo-ptm.github.io}.

![UNIMO-2](images/paper.png#pic_center)



Dependencies
---
python3.7.4\
cuda-10.1\
cudnn_v7.6\
nccl2.4.2\
java1.8
paddlepaddle-gpu==2.1.2\
pyrouge==0.1.3


Pre-trained Models
---
Similar to UNIMO, UNIMO-2 adopts large-scale text corpus, image collections and image-text aligned datasets as the pre-training data. 
We provide pre-trained UNIMO-2 models:

```
cd /path/to/model_files
wget --no-check-certificate -q https://unimo-2.bj.bcebos.com/model/UNIMO-2.tar.gz
tar -zxf UNIMO-2.tar.gz
```


Experiments
---

Our fine-tuning experiments are carried on V100 GPU. Here are the results from the UNIMO-2 model:


1 Cross-Modal  Tasks
---


### (1) Image-Text Retrieval

#### Download Flickr30k dataset:

```
cd /path/to/data
wget --no-check-certificate -q https://unimo-2.bj.bcebos.com/data/Flickr30k.tar.gz
tar -zxf Flickr30k.tar.gz
```

#### Run the following common to train and evaluate on the Flickr30k dataset:

```
bash ./script/retrieval-grounded/Flickr30k-fleet/run.sh
```

#### Evaluation Results:

Results of Image Retrieval task on Flickr30k dataset

|   Model   | R@1 |  R@5  |  R@10  |
| ----------- | ------- | ------- | ------- |
|   UNIMO-2 (zero-shot)  |  72.70 | 91.18 | 94.60  |
|   UNIMO-2 (finetuned)  |  80.14 | 95.58 | 97.75  |

Results of Text Retrieval task on Flickr30k dataset

|   Model   |  R@1  |  R@5  |  R@10  |
| ----------- | ------- | ------- | ------- |
|   UNIMO-2 (zero-shot)  |  88.46 | 96.84 | 98.92  |
|   UNIMO-2 (finetuned) | 92.01 | 99.31 | 99.51 |



### (2) Image Caption Generation

#### Download COCO Caption dataset:

```
cd /path/to/data
wget --no-check-certificate -q https://unimo-2.bj.bcebos.com/data/coco.tar.gz
tar -zxf coco.tar.gz
```

#### Download evaluation script:

```
mkdir src/eval/tasks
cd src/eval/tasks
wget --no-check-certificate -q https://unimo.bj.bcebos.com/eval_script/coco.tar.gz
tar -zxf coco.tar.gz
```

#### Run the following common to train and evaluate on the COCO Caption dataset:

```
bash ./script/img2txt-grounded/coco-oscar/run.sh
```


#### Evaluation Results:

|   Model   | BLUE4 | CIDEr |
| ----------- | ------- | ------- |
|   UNIMO-2 |  39.7  |  131.2  |



### (3) Visual Entailment
####todo



### (4) Visual Question Answering (VQA)
####todo





2 Visual Tasks
---

### (1) Image Classification
####todo

### (2) Zero-shot Image Classification
####todo



3 Textual Tasks
---

### (1) Natural Language Inference

#### Download MNLI-AX dataset:
```
cd /path/to/data
wget --no-check-certificate -q https://unimo-2.bj.bcebos.com/data/MNLI-AX.tar.gz
tar -zxf MNLI-AX.tar.gz
```

#### Run the following common to train and evaluate on the MNLI-AX dataset:

```
bash ./script/classification/MNLI-AX/run.sh
```


#### Evaluation Results:

|   Model   | Acc-(m/mm) |
| ----------- | ------- |
|   UNIMO-2  |  87.5/87.5 |




### (2) Sentiment Classification
####todo





### (3) Similarity Tasks
####todo





### (4) Linguistic Acceptability Judgments
####todo





Citation
---
If you find our paper and code useful, please cite the following paper:
```
@article{li2022unimo,
  title={UNIMO-2: End-to-End Unified Vision-Language Grounded Learning},
  author={Li, Wei and Gao, Can and Niu, Guocheng and Xiao, Xinyan and Liu, Hao and Liu, Jiachen and Wu, Hua and Wang, Haifeng},
  journal={arXiv preprint arXiv:2203.09067},
  year={2022}
}
```

Contact information
---

For help or issues using UNIMO-2, please submit a GitHub issue.

For personal communication related to UNIMO, please contact Wei Li (liwei85@baidu.com), Can Gao (gaocan01@baidu.com), Guocheng Niu (niuguocheng@baidu.com).
