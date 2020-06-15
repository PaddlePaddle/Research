#!/bin/bash 
set -xe

#export iplist=10.255.104.19,10.255.138.19,10.255.122.21
export iplist=`hostname -i` # ip of the local machine

#http_proxy
unset http_proxy
unset https_proxy
