# NACL: A General and Effective KV Cache Eviction Framework for LLM at Inference Time

## Introduction
Large Language Models (LLMs) have ignited an innovative surge of AI applications, marking a new era of exciting possibilities equipped with extended context windows. However, hosting these models is cost-prohibitive mainly due to the extensive memory consumption of KV Cache involving long-context modeling. Despite several works proposing to evict unnecessary tokens from the KV Cache, most of them rely on the biased local statistics of accumulated attention scores and report performance using unconvincing metric like perplexity on inadequate short-text evaluation. In this paper, we propose NACL, a general framework for long-context KV cache eviction that achieves more optimal and efficient eviction in a single operation during the encoding phase. Due to NACL’s efficiency, we combine more accurate attention score statistics in PROXY-TOKENS EVICTION with the diversified random eviction strategy of RANDOM EVICTION, aiming to alleviate the issue of attention bias and enhance the robustness in maintaining pivotal tokens for long-context modeling tasks. Notably, our method significantly improves the perfor- mance on short- and long-text tasks by 80% and 76% respectively, reducing KV Cache by up to 5× with over 95% performance maintenance. 

## Installation

```shell
conda create -n NACL python=3.9

conda activate NACL

wget https://paddle-whl.bj.bcebos.com/nightly/cu118/paddlepaddle-gpu/paddlepaddle_gpu-3.0.0.dev20240722-cp39-cp39-linux_x86_64.whl

pip install paddlepaddle_gpu-3.0.0.dev20240722-cp39-cp39-linux_x86_64.whl

pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html

git clone https://github.com/PaddlePaddle/Research.git

cd Research/NLP/ACL2024-NACL

pip install -r requirements.txt

```

## How to Run


### How to Download Data

Download the dataset the `data` folder. The data folder structure should be as follows.

```shell
bash scripts/download_dataset.sh
```

This will directly dump the data to `data`.

```
ACL2024-NACL
├── data
│   ├── code_debug.jsonl
│   ├── code_run.jsonl
│   ├── kv_retrieval.jsonl
│   ├── longbook_choice_eng.jsonl
│   ├── longbook_qa_chn.jsonl
│   ├── longbook_qa_eng.jsonl
│   ├── longbook_sum_eng.jsonl
│   ├── longdialogue_qa_eng.jsonl
│   ├── math_calc.jsonl
│   ├── math_find.jsonl
│   ├── number_string.jsonl
│   ├── passkey.jsonl
│   └── construct_synthetic_dataset.py
...
```

### How to Eval

In the `src` folder, execute:

#### Single GPU Eval

```shell
# eval all task
python eval_llama3.py --enable_nacl_evict --task all

# eval single task
python eval_llama3.py --enable_nacl_evict --task longbook_choice_eng
```

#### Multi GPUs Eval

```shell
# eval all task
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 eval_llama3.py --enable_nacl_evict --task all

# eval multi task
python -m paddle.distributed.launch --gpus 0,1 eval_llama3.py --enable_nacl_evict --task longbook_choice_eng,longdialogue_qa_eng
```

To test the strategy without using NACL eviction, please remove `--enable_nacl_evict`.

#### Compute Scores
```
python compute_scores.py --task all

or

python compute_scores.py --task longbook_choice_eng

```

The available tasks are:

| Task Name        | Argument to specify in `--task` |
| ---------------- | ------------------------------- |
| En.Sum           | longbook_sum_qa                 |
| En.QA            | longbook_qa_eng                 |
| En.MC            | longbook_choice_eng             |
| En.Dia           | longdialogue_qa_eng             |
| Zh.QA            | longbook_qa_chn                 |
| Code.Debug       | code_debug                      |
| Code.Run         | code_run                        |
| Math.Calc        | math_calc                       |
| Math.Find        | math_find                       |
| Retrieve.PassKey | passkey                         |
| Retrieve.Number  | number_string                   |
| Retrieve.KV      | kv_retrieval                    |



## Evaluation Result

| Task Name        | Lamma3.1 8B 128K | NACL(80% KVCache Eviction) |
| ---------------- | ---------------- | -------------------------- | 
| Retrieve.PassKey | 1.0              | 0.9457                     | 
| Retrieve.Number  | 0.9949           | 0.7474                     | 
| Retrieve.KV      | 0.592            | 0.044                      | 
| En.Sum           | 0.2761           | 0.2570                     | 
| En.QA            | 0.1303           | 0.1338                     | 
| En.MC            | 0.6637           | 0.6681                     | 
| En.Dia           | 0.17             | 0.195                      | 
| Zh.QA            | 0.1303           | 0.1330                     | 
| Code.Debug       | 0.0076           | 0.0101                     | 
| Code.Run         | 0.0125           | 0.0175                     | 
| Math.Calc        | -                | -                          | 
| Math.Find        | 0.3285           | 0.3285                     | 
| AVG              | 0.3914           | 0.3163                     | 



## Citation


```bibtex
@inproceedings{nacl2024,
      title={NACL: A General and Effective KV Cache Eviction Framework for LLM at Inference Time}, 
      author={Yilong Chen and Guoxia Wang and Junyuan Shang and  Shiyao Cui and Zhenyu Zhang and Tingwen Liu and Shuohuan Wang and Yu Sun and Dianhai Yu and Hua Wu},
      booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
      year={2024},
      publisher={Association for Computational Linguistics},
      address={Bangkok, Thailand}
}
```

## References
[$\infty $ Bench: Extending Long Context Evaluation Beyond 100K Tokens.](https://arxiv.org/abs/2402.13718)
[The Llama 3 Herd of Models](https://scontent-itm1-1.xx.fbcdn.net/v/t39.2365-6/452387774_1036916434819166_4173978747091533306_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=DTS7hDTcxZoQ7kNvgFEityk&_nc_ht=scontent-itm1-1.xx&gid=AK98To87L1-SZHQ0fCh7NFy&oh=00_AYCUYV1JtufGAbl4hVwf_rmIiU11NatzvqCsYJJ6Qn03rw&oe=66AED48D)
