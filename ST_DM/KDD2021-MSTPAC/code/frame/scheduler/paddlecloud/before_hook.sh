#!/bin/bash

pwd

chmod a+x ./frame/scheduler/paddlecloud/run_cmd.sh 
#for slurm
if [ -e ../python27-gcc482/bin/python ];then
    pydir=../python27-gcc482/bin
    #$pydir/python $pydir/pip install scikit-learn==0.20.2 --index-url=http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
    echo "slurm pass before hook" >&2
else
#for k8s
    #pip install --upgrade pip
    #pip install scikit-learn==0.20.2 --index-url=http://pip.baidu.com/pypi/simple --trusted-host pip.baidu.com
    echo "k8s pass before hook" >&2
fi

