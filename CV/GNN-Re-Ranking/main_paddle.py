import os
import argparse
import numpy as np
import paddle.fluid as fluid
from scipy import sparse
import pdb

from utils import *
from layer import vmat, qe


parser = argparse.ArgumentParser(description='GNN_Reranking')
parser.add_argument('--data_path', 
                    type=str, 
                    default='../features/market_88_test.pkl',
                    help='path to dataset')
parser.add_argument('--k1', 
                    type=int, 
                    default=26,     # Market-1501
                    # default=60,   # Veri-776
                    help='parameter k1')
parser.add_argument('--k2', 
                    type=int, 
                    default=7,      # Market-1501
                    # default=10,   # Veri-776
                    help='parameter k2')

args = parser.parse_args()

def main():   
    data = load_pickle(args.data_path)
    k1 = args.k1
    k2 = args.k2
    
    query_cam = data['query_cam']
    query_label = data['query_label']
    gallery_cam = data['gallery_cam']
    gallery_label = data['gallery_label']
          
    gallery_feature = data['gallery_f']
    query_feature = data['query_f']
    total_features = np.concatenate((query_feature,gallery_feature),axis=0)
    query_num = query_feature.shape[0]

    all_fea = fluid.layers.data(name="all_fea",shape=[total_features.shape[0], total_features.shape[1]],dtype='float32')
    x = fluid.layers.matmul(all_fea, all_fea, transpose_x=False, transpose_y=True)


    similarity, initial_rank = fluid.layers.topk(x, k=k1)
    similarity2, initial_rank2 = fluid.layers.topk(x, k=k2)
    initial_rank_fp32 = fluid.layers.cast(initial_rank, dtype='float32')
    initial_rank2_fp32 = fluid.layers.cast(initial_rank2, dtype='float32')
    V = vmat(initial_rank_fp32)

    VT = fluid.layers.transpose(V, perm=[1,0])
    similarity2 = similarity2 * similarity2

    V = V + VT
    V = qe(V, initial_rank2_fp32, similarity2)
    V_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(V), dim=1))
    V = fluid.layers.elementwise_div(V, V_norm, axis=0)
    VT = fluid.layers.transpose(V, perm=[1,0])

    V = V + VT
    V = qe(V, initial_rank2_fp32, similarity2)
    V_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(V), dim=1))
    V = fluid.layers.elementwise_div(V, V_norm, axis=0)

    score = fluid.layers.matmul(V, V, transpose_x=False, transpose_y=True)

    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() 
    exe = fluid.Executor(place) 

    exe.run(fluid.default_startup_program()) 

    outs = exe.run(
        feed={'all_fea':total_features},
        fetch_list=[score])

    cosine_similarity = np.array(outs[0])
    indices = np.argsort(-cosine_similarity[:query_num, query_num:], axis=1)
    indices = indices.reshape(query_feature.shape[0], gallery_feature.shape[0])
    evaluate_ranking_list(indices, query_label, query_cam, gallery_label, gallery_cam)
    print('error')

if __name__ == '__main__':
    main()
