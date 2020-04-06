# 事件抽取模型（基于paddlehub）

### 环境准备

- python适用版本 2.7.x（本代码测试时使用依赖见 ./requirements.txt ）
-  paddlepaddle-gpu >= 1.7.0、paddlehub >= 1.6.1
-  请转至paddlepaddle官网按需求安装对应版本的paddlepaddle

#### 依赖安装
> pip install -r ./requirements.txt

#### 模型下载&使用

```
hub install ernie_tiny==1.1.0
```

更多预训练模型参考 [PaddleHub语义模型](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel)

**使用时修改 sequence_label.py 中的 model_name = "ernie_tiny"**

### 模型训练

需要在data下放训练集(train.json)、验证集(test.json)、测试集(test.json,可用dev.json代替)、预测集(test1.json)和事件schema文件(event_schema.json)，可从[比赛官网](https://aistudio.baidu.com/aistudio/competition/detail/32?isFromCcf=true)下载

- 训练触发词识别模型

```
sh run_trigger.sh 0 ./data/ models/trigger
```

模型保存在models/trigger、预测结果保存在data/test1.json.trigger.pred

- 训练论元角色识别模型

```
sh run_role.sh 0 ./data/ models/role
```
模型保存在models/role、预测结果保存在data/test1.json.role.pred

#### 提交预测结果

把结果按照官网给定的格式提交

- 预测结果处理成提交格式

```
python data_process.py --trigger_file data/test1.json.trigger.pred --role_file data/test1.json.role.pred --schema_file data/event_schema.json --save_path data/test1_pred.json
```
整体预测结果保存在 data/test1_pred.json

- 提交结果

提交data/test1_pred.json到 [比赛官网](https://aistudio.baidu.com/aistudio/competition/detail/32?isFromCcf=true)
