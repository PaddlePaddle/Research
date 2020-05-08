#paddle env settings
export PYTHONPATH=`pwd`/PaddleDetection:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/depends/nccl_2.1.15/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/cuda/cuda-10.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/cuda/cudnn_v7.5_cuda10.1/lib64:$LD_LIBRARY_PATH
export PATH=/home/cuda/cuda-10.1/bin:$PATH
#online tracking env settings
export TRACK_DEPENDS=`pwd`/depends/
export LD_LIBRARY_PATH=/lib64:/usr/lib64:$TRACK_DEPENDS/dev_1.6_cuda10.1_cudnnv7.5/sys_lib/lib64:$TRACK_DEPENDS/adu-3rd/v2x-opencv3/output/so:$TRACK_DEPENDS/dev_1.6_cuda10.1_cudnnv7.5/third_party/install/mklml/lib:$TRACK_DEPENDS/dev_1.6_cuda10.1_cudnnv7.5/cudnn_v7.5_cuda10.1/lib64:$TRACK_DEPENDS/dev_1.6_cuda10.1_cudnnv7.5/cuda-10.1/lib64:$TRACK_DEPENDS/dev_1.6_cuda10.1_cudnnv7.5/third_party/install/tensorrt/lib:$TRACK_DEPENDS/dev_1.6_cuda10.1_cudnnv7.5/third_party/
