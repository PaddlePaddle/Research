
# 文件说明
Text2SQL 评估工具包括4个文件：
* text2sql_evaluation.py：评估脚本文件
* table.json：固定输入，为数据库描述
* gold.sql：固定输入，为样本标准答案
  - 文件每行为一条样本，由 \t 分隔为3个字段
  - 第一个字段：样本ID
  - 第二个字段：正确的 SQL
  - 第三个字段：样本对应的 db id
* pred.sql：动态输入，为每次待评估的预估结果
  - 文件每行为一条预估结果
  - 格式同上，但第三个字段的 db id 可省略

# 评估脚本用法

## 命令行调用

```
python ./text2sql_evaluation.py --table data/table.json --gold data/gold.sql --pred data/pred.sql
```

输出类似：

acc: 0.6667

## Lib 方式调用

```
from text2sql_evaluation import evaluate 
acc = evaluate('table.json', 'gold.sql', 'pred.sql')
print(acc)

```

输出类似：

0.6666666666666666

# 备注
评估脚本支持 python2、python3，但是建议使用 python3

# 感谢
评估脚本实现参考了 Spider 数据集[相关代码](https://github.com/taoyds/spider)，在此表示感谢。
