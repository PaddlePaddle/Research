export LD_LIBRARY_PATH=./lib:$( python -c 'import paddle; print(paddle.sysconfig.get_lib())'):$LD_LIBRARY_PATH
