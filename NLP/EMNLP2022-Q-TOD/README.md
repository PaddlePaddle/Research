<h2 align="center">Q-TOD: A Query-driven Task-oriented Dialogue System</h2>
<p align="center">
  <a href="https://2022.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP-2022-brightgreen"></a>
  <a href="https://arxiv.org/abs/2210.07564"><img src="https://img.shields.io/badge/Paper-PDF-blue"></a> 
</p>

## Overview

**Q-TOD** is a novel **Q**uery-driven **T**ask-**O**riented **D**ialogue system, which consists of three sequential modules: query generator, knowledge retriever, and response generator.

* Query generator extracts the essential information from the dialogue context into a concise query in an unstructured format of the natural language.
* Knowledge retriever is an off-the-shelf retrieval model, which utilizes the generated query to retrieve relevant knowledge records.
* Response generator produces a system response based on the retrieved knowledge records and the dialogue context.

<p align="center">
  <img width="80%" src="./images/architecture.png" alt="Q-TOD Architecture" />
</p>

## Requirements

* ![https://github.com/PaddlePaddle/PaddleNLP](https://img.shields.io/badge/paddlenlp-v2.4.1-blue)
* ![https://github.com/PaddlePaddle/RocketQA](https://img.shields.io/badge/rocketqa-v1.0.0-blue)
* ![https://github.com/tqdm/tqdm](https://img.shields.io/badge/tqdm-v4.46.1-blue)

## Usage

### Preparation

Prepare models and datasets.
```bash
bash ./prepare.sh
```
It downloads six fine-tuned models to `./models`:
* [Q-TOD_T5-Large_SMD](https://dialogue.bj.bcebos.com/Knover/projects/Q-TOD/Q-TOD_T5-Large_SMD.tar): the pre-trained T5-Large is fine-tuned with the SMD.
* [Q-TOD_T5-Large_CamRest](https://dialogue.bj.bcebos.com/Knover/projects/Q-TOD/Q-TOD_T5-Large_CamRest.tar): the pre-trained T5-Large is fine-tuned with the CamRest.
* [Q-TOD_T5-Large_MultiWOZ](https://dialogue.bj.bcebos.com/Knover/projects/Q-TOD/Q-TOD_T5-Large_MultiWOZ.tar): the pre-trained T5-Large is fine-tuned with the MultiWOZ.
* [Q-TOD_T5-3B_SMD](https://dialogue.bj.bcebos.com/Knover/projects/Q-TOD/Q-TOD_T5-3B_SMD.tar): the pre-trained T5-3B is fine-tuned with the SMD.
* [Q-TOD_T5-3B_CamRest](https://dialogue.bj.bcebos.com/Knover/projects/Q-TOD/Q-TOD_T5-3B_CamRest.tar): the pre-trained T5-3B is fine-tuned with the CamRest.
* [Q-TOD_T5-3B_MultiWOZ](https://dialogue.bj.bcebos.com/Knover/projects/Q-TOD/Q-TOD_T5-3B_MultiWOZ.tar): the pre-trained T5-3B is fine-tuned with the MultiWOZ.

It also downloads SMD, CamRest and MultiWOZ under the `./data`.

### Inference and Evaluation

Use fine-tuned model to infer and evaluate the test set.
```bash
bash ./infer.sh
```
After inference and evaluation, you can find results of inference and evaluation score in `./output`.

## Citation

Please cite the [paper](https://arxiv.org/abs/2210.07564) if you use Q-TOD in your work:

```bibtex
@article{tian-etal-2022-qtod,
  title={Q-TOD: A Query-driven Task-oriented Dialogue System},
  author={Tian, Xin and Lin, Yingzhan and Song, Mengfei and Bao, Siqi and Wang, Fan and He, Huang and Sun, Shuqi and Wu, Hua},
  journal={arXiv preprint arXiv:2210.07564},
  year={2022}
}
```

## Contact Information

For help or issues using Q-TOD, please submit a GitHub [issue](https://github.com/PaddlePaddle/Research/issues).
