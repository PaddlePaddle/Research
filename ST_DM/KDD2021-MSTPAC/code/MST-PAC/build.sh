#!/bin/bash
mkdir -p output
rm -rf output/*
cp -rf conf output
cp -rf datasets output
cp -rf frame output
cp -rf nets output
cp -rf test output
cp -rf utils output
cp -rf __init__.py output
cp -rf run.sh output
tar zcf paddle-frame.tar.gz output
