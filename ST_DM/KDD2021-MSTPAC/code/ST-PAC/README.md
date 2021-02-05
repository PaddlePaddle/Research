## 目录
* [环境搭建](#环境搭建)
* [使用说明](#使用说明)

## 环境搭建

1. Linux CentOS 6.3, Python 2.7, 获取PaddlePaddle v1.6.1版本以上, 请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装

2. 配置修改conf/var_sys.conf
```
fluid_bin=/home/epep/tools/paddle_release_home/python/bin/python

#gpu训练配置
cuda_lib_path=/home/epep/tools/cuda-9.0/lib64:/home/epep/tools/cudnn/cudnn_v7.3/cuda/lib64:/home/epep/tools/nccl-2.2_cuda-8.0/lib:$LD_LIBRARY_PATH
```

## 使用说明
先 cd epep 到 epep文件夹中

1. 运行 sh p3ac.sh train 原始P3AC模型训练。
   运行 sh p3ac.sh eval 原始P3AC模型预测。

2. 运行 sh st-pac.sh train 加入时间特征的P3AC模型训练。
   运行 sh st-pac.sh eval 加入时间特征的P3AC模型预测。

3. 运行 sh mst-pac.sh train meta-learning版P3AC模型fine tuning。
