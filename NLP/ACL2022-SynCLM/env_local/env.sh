#!/usr/bin/env bash
set -x
#在LD_LIBRARY_PATH中添加cuda库的路径
export LD_LIBRARY_PATH=/home/work/cuda-10.1_cudnn7.6.5/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/work/cuda-10.1_cudnn7.6.5/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#在LD_LIBRARY_PATH中添加cudnn库的路径
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7.6/cuda/lib64:$LD_LIBRARY_PATH
#需要先下载NCCL，然后在LD_LIBRARY_PATH中添加NCCL库的路径
export LD_LIBRARY_PATH=/home/work/nccl2.4.2_cuda10.1/lib:$LD_LIBRARY_PATH
#如果FLAGS_sync_nccl_allreduce为1，则会在allreduce_op_handle中调用cudaStreamSynchronize（nccl_stream），这种模式在某些情况下可以获得更好的性能
export FLAGS_sync_nccl_allreduce=1
#表示分配的显存块占GPU总可用显存大小的比例，范围[0,1]
export FLAGS_fraction_of_gpu_memory_to_use=1
#表示是否使用垃圾回收策略来优化网络的内存使用，<0表示禁用，>=0表示启用
export FLAGS_eager_delete_tensor_gb=1.0
#是否使用快速垃圾回收策略
export FLAGS_fast_eager_deletion_mode=1
#垃圾回收策略释放变量的内存大小百分比，范围为[0.0, 1.0]
export FLAGS_memory_fraction_of_eager_deletion=1

export iplist=`hostname -i`
#http_proxy
unset http_proxy
unset https_proxy
set +x
