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









