# PaddleFSL

PaddleFSL provides examples for few-shot learning (FSL), building on top of [PaddlePaddle](https://github.com/PaddlePaddle/Paddle).
Currently, it contains two backbone embedding networks, two popularly used baselines and six benchmark datasets for FSL image classification. 

## Datasets

The following benckmark FSL image classification datasets are used:

- **Omniglot** ([B. M. Lake et al., 2015](http://www.sciencemag.org/content/350/6266/1332.short)), which is downloaded from (https://github.com/brendenlake/omniglot)
- **Mini-ImageNet** ([O. Vinyals et al., 2016](https://arxiv.org/abs/1606.04080)), which is downloaded from (https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view)
- **Tiered-ImageNet** ([M. Ren et al., 2018](https://arxiv.org/abs/1803.00676)), which is downloaded from (https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view)
- **CIFAR-FS** ([L. Bertinetto et al., 2018](https://arxiv.org/abs/1805.08136)), which is downloaded from (https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view)
- **FC100** ([B. N. Oreshkin et al., 2018](https://arxiv.org/abs/1805.10123)), which is downloaded from (https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view)
- **CUB** ([W.-Y. Chen et al., 2019](https://arxiv.org/abs/1904.04232)) , which is downloaded from (https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)

We use the train / test splits as mentioned in the respective papers.

Except Tiered-ImageNet which is too large, we provide pickle files in the data folder.

## Backbones 

The following popularly used embedding architecures are used:

- **Conv4** ([O. Vinyals et al., 2016](https://arxiv.org/abs/1606.04080)), which consists of four convolutional blocks. 
- **ResNet12** ([N. Mishra et al., 2018](https://arxiv.org/abs/1707.03141)), 

## Baselines

The following baselines are provided:

- **Prototypical Networks (ProtoNet)** ([J. Snell et al., 2017](https://arxiv.org/abs/1703.05175))
- **Relation Network (RelationNet)** ([F. Sung et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf))


## Dependencies

The codes have been tested on:

* python 3.6.12
* paddlepaddle 1.8.4 
* visualdl 2.0.3


## Running Examples

### Training

~~~bash
python train.py --dataset=cifarfs --backbone='Conv4' --method='protonet' --epochs=200 --k_shot=1 --n_way=5
python train.py --dataset=miniimagenet --backbone='Conv4' --method='protonet' --epochs=200 --k_shot=5 --n_way=5
~~~

### Testing

~~~bash
python test.py --dataset=miniimagenet --backbone='Conv4' --method='protonet' --test_mode=True --test_episodes=6000 --test_model_epoch=66 --k_shot=1 --n_way=5
~~~

## Results

Using this codes, we can reproduce the reported results of respecitve papers. 

Blanks mean that public results are not avaliable.

**Omniglot**

|          |             | 1-shot-5-way(%) |      | 5-shot-5-way(%) |      |
|----------|-------------|-----------------|------|-----------------|------|
|          |             | reported        | ours | reported        | ours |
| Conv4    | ProtoNet    | 97.4|99.45 ± 0.04|99.3|99.67 ± 0.03  |
|          | RelationNet | 99.6 ± 0.2|99.23 ± 0.05|99.8± 0.1|99.83 ± 0.02  |
| ResNet12 | ProtoNet    |                 | 99.40 ± 0.04   |                 |  99.63 ± 0.03   |
|          | RelationNet |                 |      |                 |      |


**Mini-ImageNet**

|          |             | 1-shot-5-way(%) |      | 5-shot-5-way(%) |      |
|----------|-------------|-----------------|------|-----------------|------|
|          |             | reported        | ours | reported        | ours |
| Conv4    | ProtoNet    |   46.61 ± 0.78|49.75 ± 0.25|65.77 ± 0.70|66.53 ± 0.21    |
|          | RelationNet |  50.44 ± 0.82|50.04 ± 0.26|65.32 ± 0.70|65.60 ± 0.21  |
| ResNet12 | ProtoNet    |                 |  54.08 ± 0.29    |                 |   68.44 ± 0.23   |
|          | RelationNet |                 |      |                 |      |

**Tiered-ImageNet**

|          |             | 1-shot-5-way(%) |      | 5-shot-5-way(%) |      |
|----------|-------------|-----------------|------|-----------------|------|
|          |             | reported        | ours | reported        | ours |
| Conv4    | ProtoNet    |  53.31 ± 0.89|52.14 ± 0.28|72.69 ± 0.74|71.07 ± 0.24    |
|          | RelationNet |  54.48 ± 0.93|53.81± 0.29|71.32 ± 0.78|70.04 ± 0.25 |
| ResNet12 | ProtoNet    |                 |   53.51 ± 0.30   |                 |   71.04 ± 0.26   |
|          | RelationNet |                 |      |                 |      |


**CIFAR-FS**

|          |             | 1-shot-5-way(%) |      | 5-shot-5-way(%) |      |
|----------|-------------|-----------------|------|-----------------|------|
|          |             | reported        | ours | reported        | ours |
| Conv4    | ProtoNet    | 55.5 ± 0.70|55.96 ± 0.29|72.0 ± 0.60|72.27 ± 0.23   |
|          | RelationNet | 55.0 ± 1.00|55.56 ± 0.30|69.3 ± 0.80|71.45 ± 0.24     |
| ResNet12 | ProtoNet    |                 |  63.67 ± 0.31    |                 |   78.16 ± 0.23   |
|          | RelationNet |                 |      |                 |      |


**FC100**

|          |             | 1-shot-5-way(%) |      | 5-shot-5-way(%) |      |
|----------|-------------|-----------------|------|-----------------|------|
|          |             | reported        | ours | reported        | ours |
| Conv4    | ProtoNet    |  35.3 ± 0.60|36.68 ± 0.22|48.6 ± 0.60|49.45 ± 0.22  |
|          | RelationNet |                 |   36.74 ± 0.23 |                 |  47.14 ± 0.22    |
| ResNet12 | ProtoNet    |                 |35.99 ± 0.23    |                 |  46.22 ± 0.22    |
|          | RelationNet |                 |      |                 |      |


**CUB**

|          |             | 1-shot-5-way(%) |      | 5-shot-5-way(%) |      |
|----------|-------------|-----------------|------|-----------------|------|
|          |             | reported        | ours | reported        | ours |
| Conv4    | ProtoNet    | 51.31 ± 0.91|53.30 ± 0.29|70.77 ± 0.69|69.08 ± 0.22  |
|          | RelationNet | 62.45 ± 0.98|60.31 ± 0.31|76.11 ± 0.69|73.02 ± 0.23 |
| ResNet12 | ProtoNet    |                 | 60.75 ± 0.32     |                 | 71.42 ± 0.24     |
|          | RelationNet |                 |      |                 |      |


## Contributors 
This project is maintained by Yaqing Wang (`wangyaqing01@baidu.com`) and Heda Song (`v_songheda@baidu.com`) at BIL, Baidu Research. 


## Disclaimer
This project targets at fast prototyping FSL solutions for research purpose. 
Baidu is not responsible for the 3rd party's generation with these codes.


