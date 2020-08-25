# P3AC（Personalized Prefix embedding for POI Auto-Completion)

## 任务说明(Introduction)

Point of interest auto-completion (POI-AC) is a featured function in the search engine of many Web mapping services. This function keeps suggesting a dynamic list of POIs as a user types each character, and it can dramatically save the effort of typing, which is quite useful on mobile devices. Existing approaches on POI-AC for industrial use mainly adopt various learning to rank (LTR) models with handcrafted features and even historically clicked POIs are taken into account for personalization. However, these prior arts tend to reach performance bottlenecks as both heuristic features and search history of users cannot directly model personal input habits. 
In this paper, we present an end-to-end neural-based framework for POI-AC, which has been recently deployed in the search engine of Baidu Maps, one of the largest Web mapping applications with hundreds of millions monthly active users worldwide. In order to establish connections among users, their personal input habits, and correspondingly interested POIs, the proposed framework (abbr. P3AC) is composed of three components, i.e., a multi-layer Bi-LSTM network to adapt to personalized prefixes, a CNN-based network to model multi-sourced information on POIs, and a triplet ranking loss function to optimize both personalized prefix embeddings and distributed representations of POIs.
We first use large-scale real-world search logs of Baidu Maps to assess the performance of P3AC offline measured by multiple metrics, including Mean Reciprocal Rank (MRR), Success Rate (SR), and normalized Discounted Cumulative Gain (nDCG). Extensive experimental results demonstrate that it can achieve substantial improvements. Then we decide to launch it online and observe that some other critical indicators on user satisfaction, such as the average number of keystrokes and the average typing speed at keystrokes in a POI-AC session, which significantly decrease as well. In addition, we 
have released both the source codes of P3AC and the experimental data to the public for reproducibility tests.

## 安装说明(Install Guide)

### 环境准备 (Enviornment Setup)

1. paddle安装 (Install Paddle)

    本项目依赖于Paddle Fluid 1.6.1 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装

2. 下载代码 (Clone Codes)

    克隆数据集代码库到本地, 本代码依赖[Paddle-EPEP框架](https://github.com/PaddlePaddle/epep)
    ```
    git clone https://github.com/PaddlePaddle/epep.git
    cd epep
    git clone https://github.com/PaddlePaddle/Research.git
    ln -s Research/ST_DM/KDD2020-P3AC/conf/poi_qac_personalized conf/poi_qac_personalized
    ln -s Research/ST_DM/KDD2020-P3AC/datasets/poi_qac_personalized datasets/poi_qac_personalized
    ln -s Research/ST_DM/KDD2020-P3AC/nets/poi_qac_personalized nets/poi_qac_personalized
    ln -s Research/ST_DM/KDD2020-P3AC/test test/poi_qac_personalized
    cp Research/ST_DM/KDD2020-P3AC/epep_main.sh epep_main.sh
    ```

3. 环境依赖

    python版本依赖python 2.7


### 实验说明

1. 数据准备

    
    ```
    链接: https://pan.baidu.com/s/1c1Y7Cf2SN3PPX40Pm-WnHw 提取码: 3wd3
    train.dat
    dev.dat
    test.dat

    qid \t geoid \t prefix \t pos \t neg
    下载到目录：Research/ST_DM/KDD2020-P3AC/test
    ```

2. 模型训练 (Model Training)

    ```
    #如果要gpu, 编辑conf/poi_qac_personalized/poi_qac_personalized.local.conf.template, platform: local-cpu 改为 local-gpu
    sh epep_main.sh train
    ```

3. 模型评估 (Model Testing)
    ```
    #如果要gpu, 编辑conf/poi_qac_personalized/poi_qac_personalized.local.conf.template, platform: local-cpu 改为 local-gpu
    sh epep_main.sh eval

    ```

## 论文下载(Paper Download)

Please feel free to review our paper :)

https://dl.acm.org/doi/10.1145/3394486.3403318

## 引用格式(Paper Citation)

@inproceedings{10.1145/3394486.3403318,
author = {Huang, Jizhou and Wang, Haifeng and Fan, Miao and Zhuo, An and Li, Ying},
title = {Personalized Prefix Embedding for POI Auto-Completion in the Search Engine of Baidu Maps},
year = {2020},
isbn = {9781450379984},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394486.3403318},
doi = {10.1145/3394486.3403318},
pages = {2677–2685},
numpages = {9},
keywords = {point of interest, query auto-completion, poi, baidu maps, poi auto-completion, poi retrieval, personalized prefix embedding, poi embedding},
location = {Virtual Event, CA, USA},
series = {KDD '20}
}




