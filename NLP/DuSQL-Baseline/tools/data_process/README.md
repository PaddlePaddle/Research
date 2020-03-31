
# 环境要求和数据下载

代码运行需要 python 3.6.5 以上版本（低版本未经过充分测试）。

依赖的第三方python库：
* jieba

DuSQL 数据集[下载](https://dataset-bj.cdn.bcebos.com/dusql/DuSQL.tar)。解压后得到的数据包括：

* train.json: 训练集数据
* dev.json: 开发集数据
* test1.json: 测试集1数据
* db_schema.json: 数据库表格schema
* db_content.json: 数据库表格内容

# 运行代码
## 数据预处理

以原始 DuSQL 数据作为输入，输出模型训练、测试需要的数据。

数据预处理命令：
```
python3 data_process.py \
            -d path/to/train.json \
            -f True \
            -t path/to/db_schema.json \
            -c path/to/db_content.json \
            -g path/to/grammar.txt \
            -m 12 40 60 \
            -o data/model_train.json

# 参数含义：
# -d 训练/开发/测试数据
# -f 是否为训练数据(True/False)
# -t 数据库表格schema
# -c 数据库表格content
# -g grammar配置文件，即代码库根目录下的 conf/grammar.txt
# -m [最大Table数,最大Column数,最大value数]，注意跟模型训练部分保持一致
# -o 输出文件 
```
注意，上述命令的输入文件路径请根据时间情况进行修改。

## 数据后处理

以模型输出grammar rule ID 序列为输入，将grammar rule ID转换为grammar rule 进而转为最终的sql。

数据预处理命令：
```
python3 grammar2sql.py \
	-t path/to/db_schema.json \
	-d path/to/train.json file \
	-g path/to/grammar.txt \
	-m 12 40 60 \
	-r path/to/predict_rules \
	-o data/predict.sql

# 参数含义：
# -t 数据库表格schema
# -d 训练/开发/测试数据，注意需要是预处理输出的相应文件，非 DuSQL 原始数据集合
# -g grammar配置文件，即代码库根目录下的 conf/grammar.txt
# -m [最大Table数,最大Column数,最大value数]，注意跟模型训练部分保持一致
# -r 模型预测阶段输出的 rule 文件
# -o 输出文件
```

其中，“模型输出rule文件”为模型预测阶段的输出文件，其顺序必须与 `-d` 参数的 json 数据保持顺序一致。
每行为一条样本的预测输出，形式类似训练数据的 `label` 字段。

# 感谢
实现过程中参考了 [IRNet](https://github.com/microsoft/IRNet) 相关代码，在此表示感谢。

