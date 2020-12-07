# MST-PAC（Meta-Learned Spatial-Temporal POI Auto-Completion for the Search Engine at Baidu Maps)

## 任务说明(Introduction)
Point Of Interest Auto-Completion (abbr. as POI-AC) is one of the featured functions for the search engine at Baidu Maps. It can dynamically suggest a list of POI candidates within milliseconds as a user keys in each (English/Chinese/Pinyin, etc.) character. This featured function daily serves billions of search requests and can dramatically save huge amount of users' effort of typing on mobile devices. Ideally, a user may need to provide only one character and then immediately obtain her desired POI at the top of the POI list suggested by our POI-AC function. However, the state-of-the-art approach, i.e., P3AC, still has a long way to achieve this goal, although it has already taken users' profiles and their input prefixes into consideration for personalized POI suggestions. 
In this paper, we discover that some user tends to look for diverse POIs at different times or locations even if she enters the same prefix. This insight drives us to establish an end-to-end spatial-temporal POI-AC (abbr. as \textit{ST-PAC}) module to replace $P^3AC$ at Baidu Maps. To alleviate the problem of the long-tail distribution of time- \& location-specific data on POI-AC, we further propose a meta-learned \textit{ST-PAC} (abbr. as \textit{MST-PAC}) updated by an efficient MapReduce algorithm, which can significantly overcome the above issue and rapidly adapt to the cold-start POI-AC tasks with fewer examples. We sample several benchmark datasets from the large-scale search log at Baidu Maps to assess the offline performance of \textit{MST-PAC} in line with multiple metrics such as Mean Reciprocal Rank (MRR), Success Rate (SR) and normalized Discounted Cumulative Gain (nDCG). The consistent improvements in these offline metrics give us more confidence to launch this meta-learned POI-AC function online for the first time in the industry. As a result, some other critical indicators on user satisfaction, such as the average number of keystrokes (Avg. \#(KS)) in a POI-AC session, significantly decrease as well.
For now, \textit{MST-PAC} has already been deployed in production at Baidu Maps, handling billions of POI-AC tasks every day. It confirms that \textit{MST-PAC} is a practical and robust industrial solution for large-scale POI Search.
For reproducibility tests, we have also released both the source codes and the benchmark datasets for POI-AC to the research community at~\url{https://github.com/PaddlePaddle/Research/tree/master/ST_DM/KDD2021-MSTPAC/}.


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
    链接: https://pan.baidu.com/s/1c1Y7Cf2SN3PPX40Pm-WnHw 提取码: 3wd3
    train.dat
    dev.dat
    test.dat

    qid \t geoid \t prefix \t pos \t neg
    下载到目录：Research/ST_DM/KDD2020-P3AC/test
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







