#encoding=utf-8
"""
evaluation
"""
import time

import sklearn.neighbors
import numpy as np
import ml_metrics as metrics

class TopKMAP(object):
    """多模态评测工具类
    """
    def __init__(self, 
                index_info_path, 
                query_info_path, 
                query_feature_file, 
                index_feature_file):

        self.query_feature = self.load_feature(query_feature_file)
        self.index_feature = self.load_feature(index_feature_file)
        self.index_info_list = self.load_info(index_info_path)
        self.query_info_list = self.load_info(query_info_path)
        self.index_labels = self.load_labels()
        # self.make_kdt()
        self.make_nn(10)
    
    def load_info(self, file_path):
        """加载label文件
        Args:
            file_path: label文件地址
        Returns:
            data_info_list: label结果
        """
        with open(file_path, "r") as fin:
            info_str = fin.readlines()
        
        data_info_list = []
        for item in info_str:
            data_items = item.split("\t")
            data_info = {
                'source_id': data_items[0],
                'label': data_items[1],
            }
            # data_info = {
            #     'source_id': data_items[0],
            #     'label': data_items[1],
            # }
            data_info_list.append(data_info)
        
        return data_info_list

    def load_feature(self, file_path):
        """加载feature文件
        Args:
            file_path: feature文件地址
        Returns:
            features: feature结果
        """
        print(file_path)
        with open(file_path, "r") as fin:
            features_str = fin.readlines()
        
        #features_str = features_str.split("\r\n")
        
        features = {}
        for item in features_str:
            if len(item.strip().split("\t")) == 2:
                source_id, feature = item.strip().split("\t")
            else:
                source_id, feature, _ = item.strip().split("\t")
            feature = [float(e) for e in feature.split(' ') if not e is None]
            # geohash = [float(e) for e in geohash.split(' ') if not e is None]
            # geohash = list(np.array(geohash) *1.0 / np.linalg.norm(np.array(geohash)))
            # feature.extend(geohash)
            # print(len(feature))
            features[source_id] = feature
        
        return features
    
    def load_labels(self):
        """获取label对应的index序号
        Returns:
            labels: label对应的index序号
        """
        labels = {}
        cur_index = 0
        for index_item in self.index_info_list:
            if index_item['label'] not in labels:
                labels[index_item['label']] = []
            labels[index_item['label']].append(cur_index)
            cur_index += 1
        
        return labels
    
    def make_kdt(self):
        """根据index的feature构建kd-tree
        """
        feature_list = []
        for index_item in self.index_info_list:
            feature_list.append(self.index_feature[index_item['source_id']])
        feature_list = np.array(feature_list)
        self.index_kdt = sklearn.neighbors.KDTree(feature_list, leaf_size=30, metric='euclidean')

    def make_nn(self, k=20):
        """根据index的feature构建k近邻
        """
        feature_list = []
        for index_item in self.index_info_list:
            feature_list.append(self.index_feature[index_item['source_id']])
        feature_list = np.array(feature_list)
        self.neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric='euclidean', n_jobs=64).fit(feature_list)

    def calc_mapk_nn(self, k=20):
        act_res_list, predict_res_list = [], []
        feature_all = []
        for query_item in self.query_info_list:
            t1 = time.time()
            feature = np.array([self.query_feature[query_item['source_id']]])
            feature_all.append(feature[0])
            if query_item['label'] not in self.index_labels:
                act_res = []
            else:
                act_res = self.index_labels[query_item['label']]
            act_res_list.append(act_res)

        feature_all = np.array(feature_all)
        i = 0
        while i < len(feature_all.tolist()):
            t1 = time.time()
            if i + 640 > len(feature_all):
                feature_all_batch = feature_all[i:]
            else:
                feature_all_batch = feature_all[i:i + 640]
            index_arr = self.neigh.kneighbors(feature_all_batch, return_distance=False)
            predict_res_list.extend(index_arr.tolist())
            t2 = time.time()
            print(i, i + 640, t2 - t1)
            i = i + 640
        
            # predict_list = self.index_kdt.query(feature, k, return_distance=False)
            # predict_res = []
            # for predict in predict_list:
            #     for predict_index in predict:
            #         predict_res.append(predict_index)
            # if query_item['label'] not in self.index_labels:
            #     act_res = []
            # else:
            #     act_res = self.index_labels[query_item['label']]

            # # print('*' * 100)
            # # print(act_res)
            # # print(predict_res)

            # act_res_list.append(act_res)
            # predict_res_list.append(predict_res)
            # print(query_item['label'], time.time() - t1)

        
        return metrics.mapk(act_res_list, predict_res_list, k)

    def calc_mapk(self, k=20):
        """计算query的map结果
        Args:
            k: 检索的topk的限制
        Returns:
            result: topk查询结果的map
        """
        act_res_list, predict_res_list = [], []
        for query_item in self.query_info_list:
            t1 = time.time()
            feature = np.array([self.query_feature[query_item['source_id']]])
            predict_list = self.index_kdt.query(feature, k, return_distance=False)
            predict_res = []
            for predict in predict_list:
                for predict_index in predict:
                    predict_res.append(predict_index)
            if query_item['label'] not in self.index_labels:
                act_res = []
            else:
                act_res = self.index_labels[query_item['label']]

            # print('*' * 100)
            # print(act_res)
            # print(predict_res)

            act_res_list.append(act_res)
            predict_res_list.append(predict_res)
            print(query_item['label'], time.time() - t1)

        
        return metrics.mapk(act_res_list, predict_res_list, k)


if __name__ == '__main__':
    t1 = time.time()
    # query_path = '../data/test_index_query/query_bias.feature'
    # index_path = '../data/test_index_query/index_bias.feature'
    # query_path = '../data/index_query/query_bias.geohash'
    # index_path = '../data/index_query/index_bias.geohash'
    # query_path = '../data/beijing/index_query/query.merge.feature'
    # index_path = '../data/beijing/index_query/index.merge.feature'
    # query_path = '/home/map/yuwei09/code/featurefuse/data_yunpeng/query.feature'
    # index_path = '/home/map/yuwei09/code/featurefuse/data_yunpeng/index.feature'
    # query_path = '../data/beijing/index_query/query.feature'
    # index_path = '../data/beijing/index_query/index.feature'
    # query_path = '../data/index_query/query.feature.v6'
    # index_path = '../data/index_query/index.feature.v6'

    # index_info_path = "/home/map/yuwei09/code/featurefuse/data_yunpeng/index.taojin.info"
    # query_info_path = "/home/map/yuwei09/code/featurefuse/data_yunpeng/query.taojin.info"
    # query_path = '/home/map/yuwei09/code/featurefuse/data_yunpeng/query.taojin.merge.feature.v6'
    # index_path = '/home/map/yuwei09/code/featurefuse/data_yunpeng/index.taojin.merge.feature.v6'

    # index_info_path = "../data/beijing/index_query/index.info.v6"
    # query_info_path =  "../data/beijing/index_query/query.info.v6"
    # query_path =  "../data/beijing/index_query/query.v6.merge.feature"
    # index_path =  "../data/beijing/index_query/index.v6.merge.feature"

    # index_info_path = "/data3/data/imggeo/test_data/sb_match/index"
    # query_info_path =  "/data3/data/imggeo/test_data/sb_match/query"
    # query_path =  "/data3/data/imggeo/test_data/sb_match/query.feature"
    # index_path =  "/data3/data/imggeo/test_data/sb_match/index.feature"

    index_info_path = "/data3/data/imggeo/test_data/beijin/index.new"
    query_info_path =  "/data3/data/imggeo/test_data/beijin/query.new"
    query_path =  "/data3/data/imggeo/test_data/beijin/query.feature"
    index_path =  "/data3/data/imggeo/test_data/beijin/index.feature"

    eval_object = TopKMAP(index_info_path, query_info_path, query_path, index_path)
    t2 = time.time()
    print(t2 - t1)
    # print eval_object.calc_mapk(10)
    print(eval_object.calc_mapk_nn(10))
    print(time.time() - t2)


    
