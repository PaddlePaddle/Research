本项目提供视频语义标签的理解能力，其从视频的文本信息中抽取表示视频内容主旨的语义标签知识（选手可进行升级，如利用给定的知识库进行推理、融合多模信息提升标签理解效果，或生成标签等）。

## 数据处理

首先将数据整理成命名实体识别模型所需格式，并划分训练集、验证集等。可以参考[PaddleNLP中文命名实体项目](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/information_extraction/msra_ner)。

注：我们在数据处理阶段去除了未在title中出现的语义标签。

```bash
python prepare_semantic_tag.py
```

得到如下的输出文件：

```
paddle-video-semantic-tag
  |-- data
     |-- label_map.json
     |-- train.tsv
     |-- val.tsv
     |-- test.tsv
```

## 训练与验证
本模型使用了PaddleNLP模型库中的`bert-wwm-ext-chinese`模型，更多模型可参考[PaddleNLP Transformer API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/transformers.md)。

```bash
export CUDA_VISIBLE_DEVICES=0
python train_semantic_tag.py \
    --model_name_or_path bert-wwm-ext-chinese \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./data/checkpoints/semantic_tag/ \
    --device gpu
```

## 实验结果

通过2~3个epoch的训练后，在验证集上的整体效果如下：

|           | Eval  |
| :----     | :--:  |
| Precision | 0.518 |
| Recall    | 0.551 |
| F1-score  | 0.534 |

注：上表中使用的指标与线上评测使用的指标不同。上表的结果为序列标注任务中的常用的评价指标（参考[`paddlenlp.metrics.ChunkEvaluator`](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/metrics)），不考虑未在title中出现的实体；若使用线上评测系统的计算方法，验证集上的结果为`Precision=0.586, Recall=0.222, F1-score=0.322`。

## 生成语义标签结果

```bash
export CUDA_VISIBLE_DEVICES=0
python predict_semantic_tag.py \
    --model_name_or_path bert-wwm-ext-chinese \
    --max_seq_length 128 \
    --batch_size 32 \
    --device gpu \
    --init_checkpoint_path data/checkpoints/semantic_tag/model_2500.pdparams
```

生成的命名实体识别结果存储在`./predict_results/ents_results.json`
