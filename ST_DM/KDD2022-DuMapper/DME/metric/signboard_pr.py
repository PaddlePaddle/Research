#-*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
This module provide configure file management service in i18n environment.
Authors: jinwenyao(jinwenyao@baidu.com)
Date:    2018/02/08 17:23:06
"""
import os
import numpy as np
import sys
import argparse
import pickle

class SignboardPR(object):
    """
    Evaluator
    """
    def __init__(self):
        self.groundtruth_total = 0
        self.recall_total = 0
        self.match_total = 0
        self.wrong_total = 0
        self.image_total = 0
        self.bid_total = 0
        self.topone_total = 0
        self.topone_match = 0
        self.topone_exist = 0
        self.topone_wrong = 0
        self.not_have_gt = 0



    def run2(self, server_conf, embdshape):

        """
        Args:
            server_conf : 服务配置
        """
        result_dict = self.load_result(server_conf['result_path'])
        print(len(list(result_dict.keys())))


        with open(server_conf['near_imid_path'], 'rb') as f:
            near_imid_dic=pickle.load(f)
        
        with open(server_conf['other_imid_path'], 'rb') as f:
            other_imid_dic=pickle.load(f)

        distance = server_conf['distance_limit']
        model_threshold = server_conf['model_threshold']
        
        imid_list = list(near_imid_dic.keys())
        print(len(imid_list))

        self.image_total = len(imid_list)


        with open(server_conf['output_file'], 'w') as output_file:
            
            for imid in imid_list:
                if imid not in result_dict:
                    # print(imid)
                    print("{} can't be found in result".format(imid))
                    result_dict[imid] = ([0]*embdshape) #fill with 0

            for imid in imid_list:

                near_imid_list = near_imid_dic[imid]
                other_imid_list = other_imid_dic[imid]

                score_list=[]
                label_list=[]
                name_list = []

                for near_imid in near_imid_list:

                    # print(result_dict[imid])
                    # print(result_dict[near_imid])
                    # print(imid)
                    # print(near_imid)

                    score = self.get_score_V2(result_dict[imid], result_dict[near_imid])

                    if near_imid in other_imid_list:

                        label=1

                    else:

                        label=0

                    score_list.append(score)
                    label_list.append(label)
                    name_list.append(near_imid)

                # print(len(score_list))
                # print(len(label_list))

                if len(label_list)==0:

                    continue

                score_list, label_list, name_list = (list(t) for t in zip(*sorted(zip(score_list, label_list, name_list))))

                score_list=[str(x) for x in score_list]
                label_list=[str(x) for x in label_list]


                scorestr=' '.join(score_list)
                labelstr=' '.join(label_list)
                namestr = ' '.join(name_list)

                content='\t'.join([imid, scorestr, labelstr, namestr])

                output_file.write(content + '\n')

    
    def get_score_V1(self, fea, other_fea):
        """calculate distance between two features
        """
        distance_sum = 0.0   
        for f1, f2 in zip(fea, other_fea):
            distance_sum += np.linalg.norm(np.array(f1) - np.array(f2))
        return distance_sum

    def get_score_V2(self, fea, other_fea):

        """calculate distance between two features
        """

        np_fea = np.array(fea, np.float32).reshape(-1)
        np_fea = np_fea / np.linalg.norm(np_fea)

        np_other_fea = np.array(other_fea, np.float32).reshape(-1)
        np_other_fea = np_other_fea / np.linalg.norm(np_other_fea)

        score=np.sqrt(np.sum(np.square(np_fea - np_other_fea)))
        
        return score
    
    def is_match(self, fea, other_fea, model_threshold):
        """signboard is matched, change this if definition is changed
        """
        if self.get_score_V2(fea, other_fea) < model_threshold:
            return True
        return False
    def load_result(self, filename):
        """
        Args:
            filename : 结果文件路径
        Returns:
            result_dict
        """
        result_dict = dict()
        with open(filename, 'r') as result_file:
            for line in result_file.readlines():
                '''
                items = line.strip().split(' ')
                padding = 256 * 3 + 1 + 1 - len(items)
                if padding:
                    items += [0] * padding
#                assert len(items) == 256 * 3 + 1 + 1, "length not fit to description"
                image = os.path.basename(items[0])
                score = items[1]
                feature_1 = map(float, items[2:258])
                feature_2 = map(float, items[258:514])
                feature_3 = map(float, items[514:770])
                
                result_dict[image] = (feature_1, feature_2, feature_3)
                '''
                key, feature, _ = line.strip().split("\t")
                result_dict[key] = list(map(float, feature.split(' ')))
        return result_dict

    def metric(self, inp, num_thr):

        """calculate metric
        """

        def set_label(score):

            """set label to 0/1
            """
        
            if score < threshold:
                
                return 1
            
            return 0

        with open(inp, 'r') as f:
            test_list=f.readlines()

        final_recall=0
        finall_precious=0
        final_top1=0
        final_thr=0

        threshold=0.7

        for i in range(num_thr):

            TP_SUM=0
            TP_FP_SUM=0
            TP_FN_SUM=0
            TOP1_SUM=0

            for i in range(len(test_list)):
        
                
                test=test_list[i]

                test=test.strip().split('\t')

                score=list(map(float, test[1].split(' ')))
                score=list(map(set_label, score))

                label=list(map(int, test[2].split(' ')))

                TP=sum(score[0:sum(label)])
                #TP=sum(label[0:sum(score)])
                TP_FP=sum(score)
                TP_FN=sum(label)

                if score[0] == label[0]:
                    TOP1_SUM += 1

                if label[0] == 0:

                    TP_FP=TP_FP + 1
                    TP_FN=TP_FN + 1

                if label[0] == 0 and score[0] == 0:

                    TP += 1

                TP_SUM += TP
                TP_FP_SUM += TP_FP
                TP_FN_SUM += TP_FN

            recall=TP_SUM / TP_FN_SUM
            precious=TP_SUM / TP_FP_SUM
            top1=TOP1_SUM / len(test_list)
            
            if abs(precious - 0.9) < abs(finall_precious - 0.9):

                finall_precious=precious
                final_recall=recall
                final_top1=top1
                final_thr=threshold

            threshold=threshold + 0.01

        return [finall_precious, final_recall, final_top1, final_thr]

if __name__ == "__main__":
    eva = Evaluator()
    server_conf = {}
    server_conf['test_data_path'] = './taojin-test'
    server_conf['result_path'] = './feature.txt'
    server_conf['near_imid_path']='./near_imid.pkl'
    server_conf['other_imid_path']='./other_imid.pkl'
    server_conf['distance_limit'] = 80 #1000##1000
    server_conf['model_threshold'] = 1
    server_conf['output_file'] = './dist_test'
    # eva.run2(server_conf, cfg['embd_shape'])
    eva.run2(server_conf, 256)

    ##compute metric
    # p, r, top1, thr=metric('./dist_4_60000', 1)          
    # print(p, r, top1) 
