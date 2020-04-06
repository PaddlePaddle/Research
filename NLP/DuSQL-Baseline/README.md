
# 简介

## Text2SQL 任务

语义解析是一种交互式分析技术，其将用户输入的自然语言表述转成可操作执行的语义表示形式，如逻辑表达式（如一阶逻辑表示，lambda表示等）、编程语言（如SQL、python等）、数学公式等。

Text2SQL 是语义解析技术中的一类任务，让机器自动将用户输入的自然语言问题转成可与数据库交互的 SQL 查询语言，实现基于数据库的自动问答能力。

## DuSQL 数据集

在当前已存在的跨领域中文语义解析数据集中，问题类型多为基于单/多条件查询匹配的答案检索。
但在实际应用中，很多问题涉及到计算、排序比较等，当前已存在的中文数据集没有覆盖这些问题类型。

基于现有数据集问题和实际应用需求，百度NLP构建了多领域、多表数据集，覆盖了更多的问题类型。
“2020语言与智能技术竞赛（LIC 2020）”的语义解析任务基于 DuSQL 数据集进行，更多关于
LIC 2020 和 DuSQL 数据集的信息请参考 [比赛官网](https://aistudio.baidu.com/aistudio/competition/detail/30?isFromCcf=true) 。

## DuSQL 基线系统

Text2SQL 基线系统使用 PaddlePaddle 实现模型的训练和预测，并提供了效果评估和数据处理的工具。

# 环境准备
代码运行需要 Linux 主机，Python 3.6.5 以上版本和PaddlePaddle（或PaddlePaddle-GPU）1.7 以上版本。

## 推荐的环境

* 操作系统 CentOS 6.3
* Python 3.6.5
* PaddlePaddle 1.7.1

## PaddlePaddle

可根据机器情况和个人需求在 PaddlePaddle 和 PaddlePaddle-GPU 中二选一安装。
如果机器支持GPU，则建议安装GPU版本。

```
# CPU 版本
pip3 install paddlepaddle
# GPU 版本
pip3 install paddlepaddle-gpu
```

更多关于 PaddlePaddle 的安装教程、使用方法等请参考[官方文档](https://www.paddlepaddle.org.cn/#quick-start).

## 第三方 Python 库
除 PaddlePaddle 及其依赖之外，还依赖以下第三方 Python 库：
* sentencepiece 0.1.83+

可使用代码库提供的 requirements.txt 文件一键安装

```pip3 install -r requirements.txt```
 
# 数据准备
运行前需要自行下载训练、测试数据，以及预训练 ERNIE 模型（如果需要的话）。

```
# 下载模型训练、测试数据
# 得到的数据包括（位于data目录下）：
# 1. data_{version}目录: 包括训练集、开发集、测试集。其中{version}为数据版本
# 2. vocab.txt: 词表
# 3. cc.zh.300.vec.filter: 预训练的词向量文件（来自互联网，效果不做保证）
bash data/download_model_data.sh

# 下载预训练 Text2SQL 模型
# 得到的数据包括：
#   data
#   ├── trained_model：模型参数
#   │   ├── lstm: lstm encoder 版本
#   │   ├── ernie: ernie encoder 版本
#   ├── inference_model: 预测用模型
#   │   ├── lstm: lstm encoder 版本
#   │   ├── ernie: ernie encoder 版本
bash data/download_trained_model.sh

# 下载ERNIE预训练模型
# 执行完成后会得到 data/ernie1.0 目录，其中包含：
# 1. ernie_config.json: ernie 模型配置文件
# 2. vocab.txt: ernie 模型的词表
# 3. params: 预训练的模型参数
bash data/download_ernie1.0.sh
```
详细了解上述 ERNIE 数据或其模型可进一步参考<https://github.com/PaddlePaddle/ERNIE>。

# 数据处理

## 预处理
数据预处理指对原始数据进行转换、信息补充等，以适配模型训练的输入。

上述“环境准备”阶段得到的训练数据是我们预先处理好的，如需获取原始数据自行处理，可参考 `tools/data_process`
中 `README.md` 的说明自行下载数据、完成数据预处理。

## 后处理
数据后处理指将模型预估的输出结果（语法ID）转为 SQL，以便进行效果评估等工作。

具体代码和用法同样参考 `tools/data_process/README.md`。

# 运行模型

## 模型简介

该系统基于 seq2seq 模型，编码端支持 Bi-LSTM、ERNIE 两种编码方式，并支持增加额外特征辅助建模表示。
解码端使用 LSTM 建模生成过程，支持基于语法指导的解码算法。模型实现参考了 
[TranX 系统](https://github.com/pcyin/tranX) 和 [IRNet 系统](https://github.com/microsoft/IRNet)。

## 模型配置文件

模型运行必需的配置位于conf下，默认提供的配置包括：
* 训练：
  - `train_text2sql_basic.json`: 用于运行 LSTM Encoder 版训练
  - `train_text2sql_ernie.json`: 用于运行 ERNIE Encoder 版训练
* 预测：
  - `infer_text2sql_basic.json`: 用于运行 LSTM Encoder 版预测
  - `infer_text2sql_ernie.json`: 用于运行 ERNIE Encoder 版预测

下文中如无特殊说明，则上述配置统称为 config.json。

## 运行训练
### 训练 LSTM Encoder 版模型

```
bash ./run.sh script/text2sql_train.py --config conf/train_text2sql_basic.json
```

### 训练 ERNIE Encoder 版模型

```
bash ./run.sh script/text2sql_train.py --config conf/train_text2sql_ernie.json
```
注，训练 ERNIE Encoder 版模型请使用 GPU 设备，使用 CPU 训练未经充分测试且训练速度特别慢。

### 训练阶段的输出
#### 日志
训练过程会输出loss、acc相关日志，类似：
```
[39.89s]training epoch 0 steps 180: loss=17.675320, acc=0.531250
[40.56s]training epoch 0 steps 200: loss=17.108824, acc=0.718750
[54.42s]evaluate epoch 0 steps 200: loss=23.259323, acc=0.095477. best=0.095477 on epoch 0 step 200
```
其中，间隔多少steps输出一次日志、执行一次test在conf中设置：
* `trainer.train_log_step`: 每多少步打印一次 train log
* `trainer.eval_step`：每多少步执行一次test，并打印 test log

#### 保存模型
根据 conf 中 `trainer.save_model_step` 的设置，每训练 `save_model_step` 步会保存一次模型。
保存的模型包括两组：
* checkpoints: 保存当前训练的全部状态，通常用于模型热启。位于 `output/save_checkpoints/checkpoints_step_xxx`。
* inference model: 仅保存必要的模型结构和参数，通常用于预测。位于 `output/save_inference_model/inference_step_xxx`。

### 热启训练
即从之前的某个训练状态热启动本次训练，只需指定 conf 中的 `trainer.load_checkpoint` 即可：
```
"load_checkpoint": "output/save_checkpoints/checkpoints_step_100"
```

## 预测

运行预测前，请自行修改配置文件中的 `predictor.inference_model_path`。

修改配置（conf/infer_text2sql_basic.json、conf/infer_text2sql_ernie.json）

* predictor.inference_model_path: 加载训练好的模型路径
* predictor.save_predict_file: 预测结果输出路径

### 使用 LSTM Encoder 版模型预测

```
bash ./run.sh script/text2sql_infer.py --config conf/infer_text2sql_basic.json
```

### 使用 ERNIE Encoder 版模型预测

```
bash ./run.sh script/text2sql_infer.py --config conf/infer_text2sql_ernie.json
```

## 更多用法

### GPU 配置

模型当前支持 GPU 单机单卡、CPU 单机多核、CPU 单机单核运行。相关配置项如下：
* 是否使用 GPU
  - config.json 中的 `trainer.PADDLE_USE_GPU`（训练） 或 `predictor.PADDLE_USE_GPU`（预测）。
  - 0 不使用GPU
  - 1 使用GPU
* 使用第几张GPU卡
  - run.sh 中的 `CUDA_VISIBLE_DEVICES`
  - 目前仅支持单卡
* CPU 核数
  - run.sh 中的 `CPU_NUM`

另外，使用 GPU 训练时，需要指定正确的 cuda、cudnn 库。可通过
run.sh 中 `#### gpu libs ####` 下的几个配置项设置。

### 入口脚本参数

`script/text2sql_train.py` 和 `script/text2sql_infer.py` 还有其它一些参数，可通过命令参数
`-h/--help` 查看。比如：

```
$ ./run.sh script/text2sql_train.py --help

usage: text2sql_train.py [-h] --config CONFIG [--device DEVICE]
                         [--checkpoint CHECKPOINT] [--parameters PARAMETERS]
                         [--infer-model INFER_MODEL] [--save-path SAVE_PATH]
                         [--data-path DATA_PATH]
                         [--db-max-len DB_MAX_LEN DB_MAX_LEN DB_MAX_LEN]
                         [--seed SEED] [--log-file LOG_FILE] [--verbose]

text2sql main program

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file for data reader, model, trainer and so on
  --device DEVICE       device: cpu|gpu. you can specific visible cuda device
                        like gpu:0.if not setted, use the setting in json
                        config file
  --checkpoint CHECKPOINT
                        only-for-train: pre-trained model checkpoint path.
  --parameters PARAMETERS
                        only-for-train: pre-trained model parameters path.
  --infer-model INFER_MODEL
                        only-for-inference: saved inference model path.
  --save-path SAVE_PATH
                        in training process: pre-trained model parameters
                        path. in predicting process: predict result file path.
  --data-path DATA_PATH
                        train/dev/test data path root. replace <DATA_ROOT> in
                        config file
  --db-max-len DB_MAX_LEN DB_MAX_LEN DB_MAX_LEN
                        max len of db tables/columns/values. replace
                        <MAX_TABLE>, <MAX_COLUMN>, <MAX_VALUE> in config file
  --use-question-fea USE_QUESTION_FEA
                        yes|no
  --use-table-fea USE_TABLE_FEA
                        yes|no
  --use-column-fea USE_COLUMN_FEA
                        yes|no
  --use-value-fea USE_VALUE_FEA
                        yes|no
  --seed SEED           random seed. currently unsupported!
  --log-file LOG_FILE   Log file path. Default is None, and logs will be wrote
                        to stderr.
  --verbose             Runing in verbose mode, or not. Default is False.
```

命令行参数的优先级高于配置文件，即如果在命令行指定了config文件包含的参数，
则会覆盖配置文件中的相应配置。如 `--device` 会覆盖 `trainer.PADDLE_USE_GPU`，
`--checkpoint` 会覆盖 `trainer.load_checkpoint` 等。

`--data-path` 和 `--db-max-len` 两个参数的设置仅当配置文件出现相应字符串时才会生效，
程序会在解析配置之前将特定的字符串替换为命令行参数指定的值。比如配置中存在 `<MAX_TABLE>`，
则它将被替换为 `--db-max-len` 的第一个参数。


# 运行评估
本模块提供了计算预测 SQL Accuracy 的脚本，位于 `tools/evaluation` 目录。
具体用法请参考目录下的 `README.md` 文件。

使用默认的代码和配置进行模型的训练和预测，开发集效果如下：

| 模型版本      | 准确率 |
|----------     | ---    |
| LSTM Encoder  | 0.1829 |
| ERNIE Encoder | -      |

