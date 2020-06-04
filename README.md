# Research

发布基于飞桨的前沿研究工作，包括CV、NLP、KG、STDM等领域的顶会论文和比赛冠军模型。

## 目录
* [计算机视觉(Computer Vision)](#计算机视觉)
* [自然语言处理(Natrual Language Processing)](#自然语言处理)
* [知识图谱(Knowledge Graph)](#知识图谱)
* [时空数据挖掘(Spatial-Temporal Data-Mining)](#时空数据挖掘)

## 计算机视觉
| 任务类型     | 目录                                                         | 简介                                                         | 论文链接 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 车流统计 | [VehicleCounting](https://github.com/PaddlePaddle/Research/tree/master/CV/VehicleCounting/) | AICITY2020 车流统计竞赛datasetA TOP1 方案 | - |
| 车辆再识别 | [PaddleReid](https://github.com/PaddlePaddle/Research/tree/master/CV/PaddleReid) | 给定目标车辆，在检索库中检索同id车辆，支持多种特征子网络。 | |
| 车辆异常检测 | [AICity2020-Anomaly-Detection](https://github.com/PaddlePaddle/Research/tree/master/CV/AICity2020-Anomaly-Detection) |  在监控视频中检测车辆异常情况，例如车辆碰撞、失速等。| |

## 自然语言处理
| 任务类型     | 目录                                                         | 简介                                                         | 论文链接 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 中文词法分析 | [LAC(Lexical Analysis of Chinese)](https://github.com/PaddlePaddle/models/tree/develop/PaddleNLP/lexical_analysis) | 百度自主研发中文特色模型词法分析任务，集成了中文分词、词性标注和命名实体识别任务。输入是一个字符串，而输出是句子中的词边界和词性、实体类别。 | |
| 主动对话 | [DuConv](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-DuConv) | 机器根据给定知识信息主动引领对话进程完成设定的对话目标 |https://www.aclweb.org/anthology/P19-1369/|
| 语义解析 | [DuSQL-Baseline](NLP/DuSQL-Baseline) | 输入自然语言问题和相应的数据库，生成与问题对应的 SQL 查询语句，通过执行该 SQL 可得到问题的答案 | - |
| 面向推荐的对话 | [Conversational-Recommendation-BASELINE](https://github.com/PaddlePaddle/Research/tree/master/NLP/Conversational-Recommendation-BASELINE) | 融合人机对话系统和个性化推荐系统，定义新一代智能推荐技术，该系统先通过问答或闲聊收集用户兴趣和偏好，然后主动给用户推荐其感兴趣的内容，比如餐厅、美食、电影、新闻等。 | - |


## 知识图谱
| 任务类型     | 目录                                                         | 简介                                                         | 论文链接 |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 知识图谱表示学习 | [CoKE](https://github.com/PaddlePaddle/Research/tree/master/KG/CoKE) | 百度自主研发语境化知识图谱表示学习框架CoKE，在知识图谱链接预测和多步查询任务上取得学界领先效果。| [https://arxiv.org/abs/1911.02168](https://arxiv.org/abs/1911.02168) |
| 关系抽取 | [DuIE\_Baseline](https://github.com/PaddlePaddle/Research/tree/master/KG/DuIE_Baseline) | 语言与智能技术竞赛关系抽取任务DuIE 2.0基线系统，通过设计结构化标注体系，实现基于[ERNIE](https://arxiv.org/abs/1904.09223)的端到端SPO抽取模型。| - |
| 事件抽取 |[DuEE\_baseline](https://github.com/PaddlePaddle/Research/tree/master/KG/DuEE_baseline)| 语言与智能技术竞赛事件抽取任务DuEE 1.0基线系统，实现基于[ERNIE](https://arxiv.org/abs/1904.09223)+CRF的Pipeline事件抽取模型。| - |

## 时空数据挖掘


## 许可证书
此向导由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)贡献，受[Apache-2.0 license](LICENSE)许可认证。

