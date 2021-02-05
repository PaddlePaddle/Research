#!/bin/bash
num1=`awk -F "\t" '{if($1 != $2) print $0}' $1 | wc -l`
num2=`cat $1 | wc -l`
num3=`echo "scale=5;$num1/$num2"|bc`
echo $num3
