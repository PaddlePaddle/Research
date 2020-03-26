
# 数据格式说明

上述文件均为 json 格式，且整体为一个list，每个元素是个dict类型，表示一条数据。
下面对其具体字段含义进行说明。

## 问题标注数据：train.json/dev.json/test1.json

每条样本包含如下字段：

* db_id: 当前样本对应的数据库ID
* question: 原始问题文本
* question_id: 样本的唯一标识
* sql_query: 问题对应的标注 SQL
* sql: sql_query 的结构化形式，其形式亦为一个dict类型，下文对齐详细介绍。

*结构化 sql 字段说明*

我们将这个dict结构称为一个 `sql`，其包括如下字段：

```python
# select: [(agg_id, val_unit), (agg_id, val_unit), ...]
# from: {'table_units': [table_unit, table_unit, ...], 'conds': condition}
# where: condition
# groupBy: [col_unit, ...]
# orderBy: asc/desc, [(agg_id, val_unit), ...]
# having: condition
# limit: None/number
# intersect: None/sql
# except: None/sql 
# union: None/sql
```

其中的 `val_unit`, `table_unit`, `agg_id` 等说明如下：

```python
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id)
# val_unit: (calc_op, col_unit1, col_unit2)
# table_type: 'table_unit'/'sql'
# table_unit: (table_type, table_id/sql)
# cond_unit: (agg_id, cond_op, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
```

几种 op 的 id 对应关系如下：

```python
# agg_id: (none, max, min, count, sum, avg)
# calc_op: (none, -, +, \*, /)
# cond_op: (not_in, between, =, >, <, >=, <=, !=, in, like)
```

`none` 表示没有此 op。


`col_id` 和 `table_id` 分别对应下方 “数据库描述信息：db-schema.json”中的`column_names` 和 `table_names` 的下标。
即 id 是多少，就是schema相应列表中的第几个元素。

## 数据库描述信息：db-schema.json

每条数据对应一个数据库的信息，包括如下字段：

* db_id: 数据库唯一表示
* table_names: 所有表名
* column_names: 所有表的所有列名，形式为 [table_id, column_name]
* column_types: 跟 column_names 一一对应的列的数据类型
* primary_keys: 主键，对应 column_names 中的下标
* foreign_keys: 外键，对应 column_names 中的下标。每个元素包含2个值 (a, b), 表示 a 的外键是 b

## 数据库内容：db-content.json

每条数据对应一个数据库的信息，包括如下字段：

* db_id: 数据库唯一表示，与 db-schema.json 中一一对应
* tables: 各个表的内容数据，类型为 dict，key 是表名称，value 也是个dict，各字段含义如下：
  - table_name: 表名称，同外层的key
  - header: 表的各个列名
  - type: 各个列的数据类型
  - cell: 表格内容，是个嵌套的list，内层list代表一行数据，数据每个位置的值跟 header 的列相对应
