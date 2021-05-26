#! /bin/bash

#==========
set -exu
set -o pipefail
#==========

#==========preprocess JF17K & its subsets
python ./preprocessing.py --task jf17k
python ./preprocessing.py --task jf17k-3
python ./preprocessing.py --task jf17k-4

#==========preprocess WikiPeople & its subsets
python ./preprocessing.py --task wikipeople
python ./preprocessing.py --task wikipeople-
python ./preprocessing.py --task wikipeople-3
python ./preprocessing.py --task wikipeople-4
