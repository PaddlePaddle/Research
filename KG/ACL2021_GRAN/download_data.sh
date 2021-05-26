#!/bin/bash

#==========download JF17K
TASK=jf17k
mkdir -p raw/$TASK
wget --no-check-certificate https://raw.githubusercontent.com/lijp12/SIR/master/JF17k/JF17k-simple/version1/train.txt -O raw/$TASK/train.txt
wget --no-check-certificate https://raw.githubusercontent.com/lijp12/SIR/master/JF17k/JF17k-simple/version1/test.txt -O raw/$TASK/test.txt

#==========download WikiPeople
TASK=wikipeople
mkdir -p raw/$TASK
wget --no-check-certificate https://raw.githubusercontent.com/gsp2014/NaLP/master/data/WikiPeople/n-ary_train.json -O raw/$TASK/n-ary_train.json
wget --no-check-certificate https://raw.githubusercontent.com/gsp2014/NaLP/master/data/WikiPeople/n-ary_valid.json -O raw/$TASK/n-ary_valid.json
wget --no-check-certificate https://raw.githubusercontent.com/gsp2014/NaLP/master/data/WikiPeople/n-ary_test.json -O raw/$TASK/n-ary_test.json

#==========download JF17K-3, JF17K-4, WikiPeople-3, WikiPeople-4
for TASK in JF17K-3 JF17K-4 WikiPeople-3 WikiPeople-4
do
    mkdir -p raw/$TASK
    wget --no-check-certificate https://raw.githubusercontent.com/liuyuaa/GETD/master/Nary%20code/data/$TASK/train.txt -O raw/$TASK/train.txt
    wget --no-check-certificate https://raw.githubusercontent.com/liuyuaa/GETD/master/Nary%20code/data/$TASK/valid.txt -O raw/$TASK/valid.txt
    wget --no-check-certificate https://raw.githubusercontent.com/liuyuaa/GETD/master/Nary%20code/data/$TASK/test.txt -O raw/$TASK/test.txt
done
mv raw/JF17K-3 raw/jf17k-3
mv raw/JF17K-4 raw/jf17k-4
mv raw/WikiPeople-3 raw/wikipeople-3
mv raw/WikiPeople-4 raw/wikipeople-4
