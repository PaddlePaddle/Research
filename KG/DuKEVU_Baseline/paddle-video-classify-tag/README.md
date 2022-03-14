本项目基于[VideoTag飞桨大规模视频分类模型](https://github.com/PaddlePaddle/PaddleVideo/tree/application/VideoTag)。
视频分类标签模型根据视频内容在封闭的二级标签体系上进行分类，得到描述视频的分类标签。

## 数据准备

该视频分类标签模型使用TSN网络提取原始视频的向量表征。
由于该步骤比较耗时，可以直接使用我们提供的TSN视频特征。
将下载的视频特征文件解压缩，组织成如下的目录结构：

```
DuKEVU_baseline
   |-- dataset
      |-- dataset
         |-- train.json
         |-- test_a.json
      |--tsn_features_train
         |--{nid}.npy
      |--tsn_features_test_a
         |--{nid}.npy
   |-- paddle-video-classify-tag
   |-- paddle-video-semantic-tag
```

如果希望自己提取TSN视频特征，可使用如下所示的视频特征提取脚本。这里需要下载[预训练的TSN网络参数](https://videotag.bj.bcebos.com/video_tag_tsn.tar)，存放在`weights`目录下，并生成原始视频的文件路径列表。

```bash
export CUDA_VISIBLE_DEVICES=0
python tsn_extractor.py --model_name=TSN --config=./configs/tsn-single.yaml --weights=./weights/tsn.pdparams --filelist=./data/TsnExtractor.list --save_dir=./dataset/tsn_features
```

如下准备视频语义理解数据集的label集合；准备训练、验证、测试的样本列表等。

```bash
python prepare_videotag.py
```

由于数据集上有两级标签，我们分别在一级标签（level1）和二级标签（level2）的设定下进行分类实验。
在每一种设定下均需要进行训练、验证和测试的数据划分。准备过程会得到如下的输出：

```
paddle-video-classify-tag
   |-- data
      |-- level{1,2}_{train,val,test}.list
      |-- level{1,2}_label.txt
```

## 训练与验证

我们微调模型中的`AttentionLSTM`部分，需要下载[预训练的AttentionLSTM网络参数](https://videotag.bj.bcebos.com/video_tag_lstm.tar)并存放在`weights`目录下。

```
paddle-video-classify-tag
   |-- weights
      |-- attention_lstm.pdmodel
      |-- attention_lstm.pdopt
      |-- attention_lstm.pdparams
```

可以参考原代码库中的[模型微调指南](https://github.com/PaddlePaddle/PaddleVideo/blob/application/VideoTag/FineTune.md)获取更多信息。

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py --model_name=AttentionLSTM --config=./configs/attention_lstm-single-level1.yaml --pretrain=./weights/attention_lstm --save_dir=./data/checkpoints/level1 --log_interval=50
python train.py --model_name=AttentionLSTM --config=./configs/attention_lstm-single-level2.yaml --pretrain=./weights/attention_lstm --save_dir=./data/checkpoints/level2 --log_interval=50
```

## 实验结果

评测指标是F1-score（由于数据集中每个样本仅有一个正确标签，所以该指标=Hit@1=分类准确率），通过3个epoch的训练后，整体效果如下：

|          | Eval |
| :----    | :--: |
| 一级标签 | 0.62 |
| 二级标签 | 0.45 |

注：综合考虑一级标签和二级标签，验证集上的结果为`F1-score=0.54`。

## 生成分类标签结果

```bash
export CUDA_VISIBLE_DEVICES=0
# 一级标签
python predict.py --model_name=AttentionLSTM --config=./configs/attention_lstm-single-level1.yaml --weights=./data/checkpoints/level1/AttentionLSTM_epoch2.pdparams --label_file=./data/level1_label.txt --save_dir=./predict_results --save_file=level1_top1.json --log_interval=200
# 二级标签
python predict.py --model_name=AttentionLSTM --config=./configs/attention_lstm-single-level2.yaml --weights=./data/checkpoints/level2/AttentionLSTM_epoch2.pdparams --label_file=./data/level2_label.txt --save_dir=./predict_results --save_file=level2_top1.json --log_interval=200
```

生成的标签预测结果存储在`./predict_results/level{1,2}_top1.json`
