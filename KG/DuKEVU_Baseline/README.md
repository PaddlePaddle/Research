本项目为《知识增强的视频语义理解》技术评测任务的基准模型。包括两部分：1）视频分类标签模型 [paddle-video-classify-tag](./paddle-video-classify-tag)；2）视频语义标签模型 [paddle-video-semantic-tag](./paddle-video-semantic-tag)。视频分类标签模型根据视频内容在封闭的二级标签体系上进行分类，得到描述视频的分类标签；视频语义标签模型 从视频的文本信息中抽取实体语义标签（选手可进行升级，如利用给定的知识库进行推理、融合多模信息提升标签理解效果，或生成标签等）。两部分模型产出的标签结果，分别对应技术评测数据集中提供的分类标签、语义标签。

## 数据集预处理

首先下载训练和测试数据，并如下组织目录结构（注意根目录和各子目录中的`dataset`为同一个文件夹）：

```
DuKEVU_baseline
   |-- dataset
      |-- dataset
         |-- train.json
         |-- test_a.json
   |-- paddle-video-classify-tag
       |-- dataset -> ../dataset
   |-- paddle-video-semantic-tag
       |-- dataset -> ../dataset
```

## 实验环境配置

两个模块均依赖PaddlePaddle2.0环境，另外视频语义标签模型依赖paddlenlp模型库，具体可参考[PaddlePaddle官网](https://www.paddlepaddle.org.cn)和[PaddleNLP网站](https://github.com/PaddlePaddle/PaddleNLP)进行安装。

以conda实验环境为例，可按如下方式安装paddlepaddle-gpu，opencv，paddlenlp等依赖库：

```
conda create -n paddle2.0 python=3.8
conda activate paddle2.0
conda install paddlepaddle-gpu==2.0.2 cudatoolkit=10.0 -c paddle
pip install opencv-python -i https://mirror.baidu.com/pypi/simple
pip install paddlenlp==2.0.1 -i https://mirror.baidu.com/pypi/simple
pip install tqdm wget
```

## 视频语义理解模型

-  视频分类标签模型的训练、验证与测试：请参考[《视频分类标签模型》](./paddle-video-classify-tag/README.md)文档说明。

-  视频语义标签模型的训练、验证与测试：请参考[《视频语义标签模型》](./paddle-video-semantic-tag/README.md)文档说明。

## 生成预测结果提交文件

合并上述视频分类标签模型和视频语义标签模型在`test_a`评测集的测试结果，生成提交文件。

```
python generate_submission.py
```

生成的结果文件提交至评测系统后，在`test_a`评测集上的整体结果约为`0.37`。

## Acknowledgement

本项目基于[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo/tree/application/VideoTag)和[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/msra_ner)中的开源代码，特此感谢。
