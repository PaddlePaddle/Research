面向推荐的对话
===

# 简介

## 任务简介
面向推荐的对话是指集成对话系统和推荐系统的人机交互系统，该系统先通过问答或闲聊收集用户兴趣和偏好，然后主动给用户推荐其感兴趣的内容，比如餐厅、美食、电影、新闻等。

真实世界的人机交互同时涉及到多种类型的对话，比如问答、闲聊、任务型对话等。当前业界一般把多种类型的对话分开研究，这其实不符合真实的人机交互。如何自然的融合多类型对话是一个重要的挑战，为了应对这个挑战，我们提出了一个新的任务—多类型对话中的面向推荐的对话，期望系统能够主动且自然地将对话从非推荐对话（比如『问答』）引导到推荐对话，然后基于收集到的用户兴趣及用户实时反馈通过多次交互完成最终的推荐目标。

## 任务定义
给定对话相关的所有背景知识M=f<sub>1</sub>,f<sub>2</sub>,…,f<sub>n</sub> (n为知识的条数)、用户Profile (画像)P、对话场景S、第1个对话目标g<sub>1</sub>、最后2个对话目标 g<sub>L-1</sub> 和 g<sub>L</sub> 和对话目标序列的长度(对话目标的个数)L（L≥3）。要求参赛系统先预测对话目标序列中其他目标，再输出符合当前对话历史H=u<sub>1</sub>,u<sub>2</sub>,…,u<sub>t-1</sub>(1＜t≤m，m为对话的utterance个数)和当前对话目标序列 G=g<sub>1</sub>、g<sub>2</sub>、g<sub>3</sub>…g<sub>q-1</sub>(1＜q≤L，L为目标序列长度)的机器（参赛模型只需模拟机器角色即可）回复u<sub>t</sub> ，同时使得对话自然流畅、信息丰富。

输入/输出：

输入：第一个对话目标g<sub>1</sub>、倒数第二个对话目标g<sub>L-1</sub>、知识信息M、用户Profile (画像)P、对话场景S、对话目标序列的长度L和对话历史H

输出： 目标序列中其他目标 g<sub>2</sub>、g<sub>3</sub>… g<sub>L-2</sub>；同时符合对话历史和对话目标序列，且自然流畅、信息丰富的机器回复u<sub>t</sub>

## 数据集
数据包括：用户Profile、对话相关的知识、对话的目标序列、对话场景和对话内容等。用户Profile包括用户的一些个人信息、领域偏好和实体偏好等。对话知识信息来源于明星、电影、音乐、新闻、美食、POI、天气等领域的有聊天价值的知识信息，如明星领域的个人信息、代表作、成就、评价等，电影领域的票房、主演、导演、评价等，以三元组SPO的形式组织。对话的目标序列包括3-5个对话目标，每个对话目标包括两部分：对话类型和对话话题。对话类型包括：QA、面向推荐的对话、任务型对话和闲聊。对话话题为明星、电影、音乐等领域的实体，或新闻等有聊天价值的知识信息。对话场景包括聊天的时间、地点和主题等。训练集包括约10万轮对话，开发集包括约1.5万轮对话，第一批测试集包含约5000个样本，第二批测试集包括约20000个样本，每个对话平均7-8轮。

具体数据样例及说明见[竞赛官网](https://aistudio.baidu.com/aistudio/competition/detail/29)。

## 基线系统(建议比赛使用生成模型)
我们同时提供检索模型和生成模型。所有模型都是基于百度深度学习框架[PaddlePaddle](http://paddlepaddle.org/)实现的。基线系统包括三个大功能：
1.goal_planning，目标规划，根据对话历史、知识库、用户Profile、对话目标历史等内容，为对话规划对话目标。<br>
2.retrieval_model，检索模型，基于第1步规划的对话目标，根据对话历史、知识库、用户Profile、对话目标历史等内容，检索出对话的回复。<br>
3.generative_model，生成模型，基于第1步规划的对话目标，根据对话历史、知识库、用户Profile、对话目标历史等内容，生成对话的回复。<br>

两个模型的效果如下：

| 基线系统 | F1/BLEU2 |DISTINCT2 |
| ------------- | ------------ | ------------ |
| 检索模型 | 34.73/0.230 | 0.189 |
| 生成模型 | 38.17/0.221 | 0.056 |



# 快速开始

## 安装
### 环境依赖
经测试，基线系统可在以下环境正常运行

 * 系统：CentOS 6.3, cuda 9.0, CuDNN 7.0
 * python 2.7
 * PaddlePaddle 1.6.1


### 安装代码
克隆工具集代码库到本地

```shell
git clone https://github.com/PaddlePaddle/Research.git
cd Research/NLP/conversational-recommendation-BASELINE/
```

### 安装第三方依赖
```
conda create -n Dialog pip python=2.7
source activate Dialog
pip install -r requirements.txt
```

## 运行
### 下载数据集
按[竞赛官网](https://aistudio.baidu.com/aistudio/competition/detail/29)的说明下载数据集。

### goal_planning训练和测试

#### 预处理数据
按官网说明下载数据，并放到`goal_planning/origin_data/resource/`目录下，再生成模型训练所需数据。

```
cd goal_planning/model
python3 process_data_for_goal_planning.py

cd ../data_generator
python3 data_generator.py
python3 train_generator.py
```

#### 训练：
```
cd goal_planning/model
python paddle_binary_lstm.py，评估当前goal是否完成
python paddle_astar_goal.py，如果当前goal完成，预测下一个goal的type
python paddle_astar_kg.py，如果当前goal完成，预测下一个goal的topic
```

#### 测试：

```
cd goal_planning/model
python goal_planning.py，完整的goal planning
```


### retrieval_model训练和测试

#### 预处理数据

按官网说明下载数据，并放到`data/resource/`目录下，处理成和`data/resource/train/dev/test.txt`相同的数据格式: 

```
./data/resource/train.txt
./data/resource/dev.txt
./data/resource/test.txt
```

#### 训练模型

```bash
cd retrieval_model
sh run_train.sh match_kn_gene
```

#### 测试模型

```bash
cd retrieval_model
sh run_test.sh match_kn_gene
```

### generative_model训练和测试

#### 预处理数据
我们做了简单处理，直接拉取代码即可跑出结果。如想进一步提升效果，需要基于`goal planning`预测出更好的goal，然后替换数据集中的属性`goal`，我们通过实验发现加入goal能比较明显的提升效果。

注意：`generative_model/data/sgns.weibo.300d.txt`是生成模型所需要的embedding文件，因原始文件较大，只放了100行供参赛者了解文件格式。参赛者可自行训练更好的embedding。`data/resource/train/dev/test.txt`包含少量数据，需要替换成完整数据。

#### 训练模型

```bash
cd generative_model
sh run_train.sh
```

#### 测试模型

```bash
cd generative_model
sh run_test.sh
```


# 目录结构

```text
.
├── requirements.txt                  # 第三方依赖
├── README.md                         # 本文档
└── conversational-recommendation     # 源码
├── generative_model                  # 生成模型
│   ├── data                          # 数据
│   ├── models                        # 默认模型保存路径
│   ├── network.py                    # 模型配置、训练和测试
│   ├── output                        # 默认输出路径
│   ├── run_test.sh                   # 测试脚本
│   ├── run_train.sh                  # 训练脚本
│   ├── source                        # 模型的实现
│   └── tools                         # 工具
├── goal_planning                     # 对话目标规划
│   ├── logs                          # 保存的log
│   ├── data_generater                # 生成训练所需数据
│   ├── process_data                  # 处理后的数据     
│   ├── model                         # 模型
│   ├── model_state                   # 默认模型保存路径
│   ├── train_data                    # 转换为模型所需数据
│   └── origin_data                   # 原始数据
└── retrieval_model                   # 检索模型
    ├── args.py                       # 参数配置
    ├── data                          # 数据
    ├── dict                          # dict
    ├── interact.py                   # 人工评估
    ├── models                        # 默认模型保存路径
    ├── output                        # 默认输出路径
    ├── predict.py                    # 模型测试
    ├── run_predict.sh                # 测试脚本
    ├── run_train.sh                  # 训练脚本
    ├── source                        # 模型实现
    ├── tools                         # 工具
    └── train.py                      # 模型训练

```


# 其他
## 如何贡献代码

我们欢迎开发者向基线系统贡献代码。如果您开发了新功能，发现了bug……欢迎提交Pull request与issue到Github。
