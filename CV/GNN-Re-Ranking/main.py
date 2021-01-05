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

    X_u = fluid.layers.data(name="all_fea",shape=[total_features.shape[0], total_features.shape[1]],dtype='float32')
    original_score = fluid.layers.matmul(X_u, X_u, transpose_x=False, transpose_y=True)

    _, initial_rank_k1 = fluid.layers.topk(original_score, k=k1)
    S, initial_rank_k2 = fluid.layers.topk(original_score, k=k2)

    initial_rank_k1_fp32 = fluid.layers.cast(initial_rank_k1, dtype='float32')
    initial_rank_k2_fp32 = fluid.layers.cast(initial_rank_k2, dtype='float32')

    # stage 1
    A = vmat(initial_rank_k1_fp32)
    S = S * S

    # stage 2
    if k2 != 1:      
        for i in range(2):
            AT = fluid.layers.transpose(A, perm=[1,0])
            A = A + AT
            A = qe(A, initial_rank_k2_fp32, S)
            A_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(A), dim=1))
            A = fluid.layers.elementwise_div(A, A_norm, axis=0)

    score = fluid.layers.matmul(A, A, transpose_x=False, transpose_y=True)

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

if __name__ == '__main__':
    main()
