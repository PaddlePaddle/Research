## DuIE2.0 Baseline System  
### Abstract  
This is the baseline system developed for DuIE 2.0 relation extraction task.
Different from [DuIE 1.0](http://lic2019.ccf.org.cn/kg), DuIE 2.0 is more inclined to colloquial language, and further introduces **complex relations** which entails multiple objects in one single SPO.
This baseline system is built upon a SOTA pre-trained language model [ERNIE](https://arxiv.org/abs/1904.09223).
We design a structured **tagging strategy** to directly fine-tune ERNIE, through which multiple, overlapped SPOs can be extracted in **a single pass**.  
The system was originally created for the [lic2020 competition](https://aistudio.baidu.com/aistudio/competition/detail/31?isFromCcf=true) and was first released in 2020.03.
And this is our new implementation with paddlepaddle2.0.
### Tagging Strategy  
Our tagging strategy is designed to discover multiple, overlapped SPOs in the DuIE 2.0 task.
Based on the classic 'BIO' tagging scheme, we assign tags (also known as labels) to each token to indicate its position in an entity span.
The only difference lies in that a "B" tag here is further distinguished by different predicates and subject/object dichotomy.
Suppose there are N predicates. Then a "B" tag should be like "B-predicate-subject" or "B-predicate-object",
which results in 2*N **mutually exclusive** "B" tags.
After tagging, we treat the task as token-level multi-label classification, with a total of (2*N+2) labels (2 for the “I” and “O” tags).  
Below is a visual illustration of our tagging strategy:
<div  align="center">  
<img src="./tagging_strategy.png" width = "550" height = "420" alt="Tagging Strategy" align=center />
</div>

For **complex relations** in the DuIE 2.0 task, we simply treat affiliated objects as independent instances (SPOs) which share the same subject.
Anything else besides the tagging strategy is implemented in the most straightforward way. The model input is:
 <CLS> *input text* <SEP>, and the final hidden states are directly projected into classification probabilities.

### Environments  
Python3 + paddlepaddle2.0.0rc1 + paddlenlp2.0.0rc1.
The code is tested on a single P40 GPU, with CUDA version=10.2, GPU Driver Version = 440.33.01.

### Download Dataset
Please download the dataset from the **competition website**, then unzip files into `./data/` and rename them to `train_data.json`, `dev_data.json` and `test_data.json`.

### Training  
```
sh train.sh
```
By default the checkpoints will be saved into `./checkpoints/` and evaluated with p/r/f1 metrics.
GPU ID can be specified in the script.
Batchsize of 8 and max_seq_len of 128 will consume GPU memory for about 4GB.

The best performance on dev set is **69.82** f1 score, train 140,000 steps with learning rate = 2e-5 and batch size = 8.
### Prediction  
Specify your checkpoints dir and dataset file in the script and then run:
```
sh predict.sh
```
Batchsize of 8 and max_seq_len of 512 will take 2 hours and 10GB GPU memory consumption.

You shall get the prediction and zipped prediction file as in `./data/predictions.json` and `./data/predictions.json.zip`.

The prediction file is in the same format as the original dataset (required for final official evaluation).

 - You can run prediction on `dev_data.json` and evaluate using official script as:
```
python re_official_evaluation.py --golden_file=dev_data.json  --predict_file=predicitons.json.zip [--alias_file alias_dict]
```
Precision, Recall and F1 scores are used as the official evaluation metrics to measure the performance of participating systems. Alias file lists entities with more than one correct mentions, it will be considered in final competition evaluation, and we do not provide it here.

 - Or you can run prediciton on `test_data.json` and submit your zipped prediction file to our competition. 