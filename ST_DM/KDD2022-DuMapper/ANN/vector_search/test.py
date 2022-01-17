# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from interface import Graph_Index

from tqdm import tqdm

# with open("/data3/BestModel/MMF/brand/gallery.feature", 'r') as f:
#     index_feature = f.readlines()

# with open("/data3/data/brand/test_other/gallery", 'r') as f:
#     index_label = f.readlines()
#     name2label = {}
#     for item in index_label:
#         name, label = item.strip().split('\t')
#         name2label[name] = label

# index_vectors = []
# index_docs = []
# for item in index_feature:
#     name, feature, _ = item.strip().split("\t")
#     feature = [float(x) for x in feature.split(' ')]
#     index_vectors.append(feature)
#     index_docs.append(name2label[name])

# index_vectors = np.array(index_vectors).astype(np.float32)
# indexer = Graph_Index(dist_type="IP") 
# indexer.build(gallery_vectors=index_vectors, gallery_docs=index_docs, pq_size=100, index_path='brand_gallery')


with open("/data3/BestModel/MMF/brand/query.feature.all", 'r') as f:
    query_feature = f.readlines()

with open("/data3/data/brand/test_other/query", 'r') as f:
    index_label = f.readlines()
    name2label = {}
    for item in index_label:
        name, label = item.strip().split('\t')
        name2label[name] = label

query_vectors = []
query_docs = []
name_lst = []
for item in query_feature:
    if len(item.strip().split("\t")) == 2:
        name, feature= item.strip().split("\t")
    else:
        name, feature, _ = item.strip().split("\t")
    feature = [float(x) for x in feature.split(' ')]
    query_vectors.append(feature)
    query_docs.append(name2label[name])
    name_lst.append(name)

query_vectors = np.array(query_vectors).astype(np.float32)
#index_vectors = index_vectors / index_norm
print(query_vectors.shape)

indexer = Graph_Index(dist_type="IP") 
indexer.load(index_path="brand_gallery")
query_vector = query_vectors[0]
scores, docs = indexer.search(query=query_vector, return_k=1, search_budget=100)
print(scores, docs)
with open('result_new', 'w') as f:
    for i in tqdm(range(query_vectors.shape[0])):
        name = name_lst[i]
        query_vector = query_vectors[i]
        tru_label = query_docs[i]
        scores, docs = indexer.search(query=query_vector, return_k=1, search_budget=100)
        score = scores[0]
        pre_label = docs[0]
        info = '\t'.join([name, str(tru_label), str(pre_label), str(score)]) + '\n'
        f.write(info)

