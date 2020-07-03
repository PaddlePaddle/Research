# P3AC（Personalized Prefix embedding for POI Auto-Completion)

## 任务说明(Introduction)

TODO

## 安装说明(Install Guide)

### 环境准备 (Enviornment Setup)

1. paddle安装 (Install Paddle)

    本项目依赖于Paddle Fluid 1.6.1 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装

2. 下载代码 (Clone Codes)

    克隆数据集代码库到本地, 本代码依赖[Paddle-EPEP框架](https://github.com/PaddlePaddle/epep)
    ```
    git clone https://github.com/PaddlePaddle/epep.git
    cd epep
    git clone https://github.com/PaddlePaddle/Research.git
    ln -s Research/ST_DM/KDD2020-P3AC/conf/poi_qac_personalized conf/poi_qac_personalized
    ln -s Research/ST_DM/KDD2020-P3AC/datasets/poi_qac_personalized datasets/poi_qac_personalized
    ln -s Research/ST_DM/KDD2020-P3AC/nets/poi_qac_personalized nets/poi_qac_personalized
    ln -s Research/ST_DM/KDD2020-P3AC/test test/poi_qac_personalized
    cp Research/ST_DM/KDD2020-P3AC/epep_main.sh epep_main.sh
    ```

3. 环境依赖

    python版本依赖python 2.7


### 实验说明

1. 数据准备

    
    ```
    https://github.com/anpark/Research/blob/master/ST_DM/KDD2020-P3AC/test
    train.dat
    dev.dat
    test.dat
    ```

2. 模型训练 (Model Training)

    ```
    #如果要gpu, 编辑conf/poi_qac_personalized/poi_qac_personalized.local.conf.template, platform: local-cpu 改为 local-gpu
    sh epep_main.sh train
    ```

3. 模型评估 (Model Testing)
    ```
    #如果要gpu, 编辑conf/poi_qac_personalized/poi_qac_personalized.local.conf.template, platform: local-cpu 改为 local-gpu
    sh epep_main.sh eval

    ```

## 论文下载(Paper Download)

Please feel free to review our paper :)

TODO

## 引用格式(Paper Citation)

TODO


