### AICITY2020-track1
In this repo, we include the 1st Place submission to [Aicity Challenge](https://www.aicitychallenge.org/) 2020 vehicle counting track(Baidu submission).

Zhongji Liu, Wei Zhang, Xu Gao, Hao Meng, Zhan Xue, Xiao Tan, Xiaoxing Zhu, Hongwu Zhang, Shilei Wen, and Errui Ding. Robust movement-specific vehicle counting at crowded intersections. In Proc. CVPR Workshops, Seattle, WA, USA, 2020.
Contact: zhangwei99@baidu.com, liuzhongji@baidu.com. Any questions or discussion are welcome!

#### Performance:

AICITY2020 Challange Track1 Leaderboard

| TeamName        | Score  |
| --------------- | ------ |
| **Baidu(Ours)** | 0.9389 |
| ENGIE           | 0.9346 |
| CMU             | 0.9292 |
| BUT             | 0.8829 |
| KISTI           | 0.8540 |

### Environment

This repo is developed under the following configurations:

+ Hareware: Centos 7, 4 NVIDIA P4 GPUs

+ Software: Python=3.6.7 GCC 5.4.0 paddle=1.7.0 CUDA=9.0

We provide a docker image for official test for track-1 datasetB. 

Download via https://bj.bcebos.com/v1/baixue/liuzhongji/aicity2020_task1_counting.tar 

We also provide our label of datasetA for fineturning vehicle detector.

Download via Baidu Cloud: [https://pan.baidu.com/s/1o8QMCubHCO6dTTSELCcNuA  passwd: r08r]

where:

train.tar -- selected frames from datasetA

1584811984_train_final.json -- label 

faster_rcnnn_r50_track1.tar -- configurations for fineturning


####How to run:

+ load the docker image and create a docker container(or prepare paddle environment follow [this](https://github.com/PaddlePaddle/Paddle))

`docker load < baidu_aicity2020_track1.tar`

`nvidia-docker run  --name aicity2020_task1 --shm-size 16G -it aicity2020:track1`

`docker attach aicity2020_track1`

the code is in the path: `/home/task1_code`, the cuda env file is in the path: `/home/cuda`

+ prepare the dataset( copy the whole dataset of "AIC20_track1" to `/home/task1_code*` )
+ prepare the run env:   `cd /home/task1_code && source set_env.sh`
+ modify the path of list_video_id.txt in: (1) run_pipeline.sh ; (2) vehicle_counting/counting.py

+ **run the whole pipline:  `sh run_pipeline.sh`**

the steps and related folders included in the whole pipeline:

1. **extract_frames.sh: **extract images from videos in dataset like "AIC20_track"
2. **run_detection.sh: ** /home/task1_code/PaddleDetection is the folder where we run detection model, the origional /tools/infer.py, which you can find at [paddle detection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.2/tools) is modified to save detection results in txt format. The detection results will be saved in /home/task1_code/det_results/
3. **run_tracking.sh: ** /home/task1_code/tracking is the folder where we run online tracking method, the tracking results will be saved to txt files in /home/task1_code/track_results/
4. **run_counting.sh: **/home/task1_code/vehicle_counting is the folder wher we run counting method, the final counting results will be saved to txt files in /home/task1_code/vehicle_counting_results/

The related repos:

[paddle detection] https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.2

[py-hausdorff] https://github.com/mavillan/py-hausdorff
