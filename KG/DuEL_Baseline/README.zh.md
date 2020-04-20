# CCKS2020 EL Baseline System
## 摘要
为CCKS 2020开发的实体链指基线系统，在CCKS 2019面向中文短文本的实体链指任务的基础上进行了拓展与改进。有关数据集和任务的详细信息，请访问我们[比赛](https://biendata.com/competition/ccks_2020_el/)的官方网站。
## 思路
DuEL模型是一个多任务模型，包含两个任务：候选实体排名和提及类型的预测。

候选实体排序任务使用pairwise模型。在训练阶段，将query和候选实体描述分别输入Ernie网络以query表示和候选实体表示。
之后，合并query表示和候选实体表示并经过MLP网络进行打分。这里使用的损失函数是rank_loss。

Mention 类型预测任务使用分类模型。在训练阶段，将query输入到Ernie网络中以获取表示。
之后，query表示向量通过MLP获得mention类型。这里的损失函数是分类损失。

最后，我们将两个损失合并后，进行多任务训练。

在预测阶段，我们使用预测的mention类型来验证候选实体。

注，在训练和预测阶段，我们将NIL视为实体ID，详细信息见代码。

<img src="strategy.png" width="60%" height="60%" align=center>

我们将一些模型输入数据的示例放在dir [./data/generated/](./data/generated)下，该数据可以通过代码[./data/data_process.py](./data/)生成。
## 环境
Python2 + Paddle Fluid 1.5 (请在脚本中确认您的Python路径)。

需求包在./requirements.txt中列出

代码在单个P40 GPU上进行了测试，CUDA版本=10.0
## 下载数据
请从[竞赛网站](https://biendata.com/competition/ccks_2020_el/data/)上下载数据，解压后放在./data/basic_data/目录

解压后,目录./data/basic_data/ 中包含文件: 
    
    dev.json
    kb.json
    test.json
    train.json
    eval.py
    README
    CCKS 2020 Entity Linking License.docx
    
## 下载预训练的Ernie模型
下载ERNIE1.0 Base模型，并将其解压到./pretrained_model/

cd ./pretrained_mdoel/

wget --no-check-certificate https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz

tar -zxvf ERNIE_1.0_max-len-512.tar.gz

解压后,路径./pretrained_model/ERNIE_1.0_max-len-512 中包含文件：ernie_config.json、params、vocab.txt
## 数据格式转换
生成的数据在./data/generated/目录中

cd ./data/

python data_process.py
## 训练
sh ./script/train.sh

默认情况在，模型将保存到./checkpoints/

训练过程中会打印准确率和f1

训练和预测前注意调整python路径和数据集路径

建议使用10000条以下的数据作为验证集，以免验证集过大导致整体耗时增加
## 预测
调整预测脚本中的模型路径，运行：

sh ./script/predict.sh

预测结果将以与原始数据集的相同的格式写入json文件（与最终官方评估格式相同）。预测结果路径为./data/generated/test_pred.json
## 评估
修改评估文件中的预测结果路径（使用dev数据集进行本地评估，需调整预测步骤的数据集和数据生成过程的is_train参数），运行：

python ./data/eval.py

提交生成的test_pred.json到竞赛平台进行测试集评估。

## Copyright and License 

Copyright 2020 Baidu.com, Inc. All Rights Reserved

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. You may otain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions 
and limitations under the License.
## 附录

数据集包含24种实体类型，详情见下表

|Type| 中文名 |
|:---|:---|
|Event|	事件活动|
|Person|	人物|
|Work|	作品|
|Location|	区域场所|
|Time&Calendar|	时间历法|
|Brand|	品牌|
|Natural&Geography|	自然地理|
|Game|	游戏|
|Biological|	生物|
|Medicine|	药物|
|Food|	食物|
|Software|	软件|
|Vehicle|	车辆|
|Website|	网站平台|
|Disease&Symptom|	疾病症状|
|Organization|	组织机构|
|Awards|奖项|
|Education|	教育|
|Culture|	文化|
|Constellation|	星座|
|Law&Regulation|	法律法规|
|VirtualThings|	虚拟事物|
|Diagnosis&Treatment|	诊断治疗方法|
|Other|	其他|