from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import OrderedDict


def recall_at_k(score_matrix, text2img, img2texts):
    assert score_matrix.shape[0] == len(text2img) * len(img2texts)
    cur_img, cur_cap = score_matrix[:, 1], score_matrix[:, 2]
    img_len, cap_len = len(np.unique(cur_img)), len(np.unique(cur_cap))
    
    cur_img_sort = np.reshape(np.argsort(cur_img), [-1, cap_len])
    cur_cap_sort = np.reshape(np.argsort(cur_cap), [-1, img_len])
    i2c = np.take(score_matrix, cur_img_sort, axis=0) # img_len x cap_len x 3
    c2i = np.take(score_matrix, cur_cap_sort, axis=0) # cap_len x img_len x 3

    def get_recall_k(scores, idx, label_dict):
        '''
        scores: sample x len x 5
        idx: 1 means text retrieval(i2c), 2 means image retrieval(c2i)
        '''
        cand_idx_dict = {1: 2, 2: 1}
        cand_idx = cand_idx_dict[idx]
        tot = scores.shape[0]
        r1, r5, r10, rank_tot = 0, 0, 0, 0

        for i in range(tot):
            score_mat = scores[i]
            cur_ids = score_mat[0][idx]
            ans_ids = label_dict[cur_ids] # when idx is 1, type is list. idx is 2, type is int

            score = score_mat[:, 0]
            score_sort = np.argsort(score)[::-1]
            cand_ans = np.take(score_mat[:, cand_idx], score_sort, axis=0)
            cand_ans = cand_ans.astype(np.int64)

            if isinstance(ans_ids, list):
                rank = min([np.where(cand_ans == ans)[0] for ans in ans_ids])
            elif isinstance(ans_ids, int):
                rank = np.where(cand_ans == ans_ids)[0]
            else:
                raise ValueError('type error')
            if rank < 1:
                r1 += 1.0
            if rank < 5:
                r5 += 1.0
            if rank < 10:
                r10 += 1.0
            rank_tot += (rank + 1)
        ret = {
                'recall@1': float(r1)/tot,
                'recall@5': float(r5)/tot,
                'recall@10': float(r10)/tot,
                'avg_rank': float(rank_tot)/tot
              }
        return ret

    cap_retrieval_recall = get_recall_k(i2c, 1, img2texts)
    img_retrieval_recall = get_recall_k(c2i, 2, text2img)

    ret = OrderedDict()
    ret['img_avg_rank'] = img_retrieval_recall['avg_rank']
    ret['cap_avg_rank'] = cap_retrieval_recall['avg_rank']

    ret['img_recall@1'] = img_retrieval_recall['recall@1']
    ret['img_recall@5'] = img_retrieval_recall['recall@5']
    ret['img_recall@10'] = img_retrieval_recall['recall@10']

    ret['cap_recall@1'] = cap_retrieval_recall['recall@1']
    ret['cap_recall@5'] = cap_retrieval_recall['recall@5']
    ret['cap_recall@10'] = cap_retrieval_recall['recall@10']

    ret['avg_img_recall'] = (img_retrieval_recall['recall@1'] + img_retrieval_recall['recall@5'] + img_retrieval_recall['recall@10']) /3
    ret['avg_cap_recall'] = (cap_retrieval_recall['recall@1'] + cap_retrieval_recall['recall@5'] + cap_retrieval_recall['recall@10']) /3

    ret['avg_recall@1'] = (img_retrieval_recall['recall@1'] + cap_retrieval_recall['recall@1']) /2
    ret['avg_recall@5'] = (img_retrieval_recall['recall@5'] + cap_retrieval_recall['recall@5']) /2
    ret['avg_recall@10'] = (img_retrieval_recall['recall@10'] + cap_retrieval_recall['recall@10']) /2

    ret['key_eval'] = "avg_recall@1"
    return ret


def recall_at_k_caption(score_matrix, img2texts, cap_len=100):
    cur_img = score_matrix[:, 1]
    cur_img_sort = np.reshape(np.argsort(cur_img), [-1, cap_len])
    i2c = np.take(score_matrix, cur_img_sort, axis=0) # img_len x cap_len x 3

    def get_recall_k(scores, idx, label_dict):
        '''
        scores: sample x len x 5
        idx: 1 means text retrieval(i2c), 2 means image retrieval(c2i)
        '''
        cand_idx_dict = {1: 2, 2: 1}
        cand_idx = cand_idx_dict[idx]
        tot = scores.shape[0]
        r1, r5, r10, rank_tot = 0, 0, 0, 0

        for i in range(tot):
            score_mat = scores[i]
            cur_ids = score_mat[0][idx]
            ans_ids = label_dict[cur_ids] # when idx is 1, type is list. idx is 2, type is int

            score = score_mat[:, 0]
            score_sort = np.argsort(score)[::-1]
            cand_ans = np.take(score_mat[:, cand_idx], score_sort, axis=0)
            cand_ans = cand_ans.astype(np.int64)

            if isinstance(ans_ids, list):
                rank = min([np.where(cand_ans == ans)[0] for ans in ans_ids])
            elif isinstance(ans_ids, int):
                rank = np.where(cand_ans == ans_ids)[0]
            else:
                raise ValueError('type error')
            if rank < 1:
                r1 += 1.0
            if rank < 5:
                r5 += 1.0
            if rank < 10:
                r10 += 1.0
            rank_tot += (rank + 1)
        ret = {
                'recall@1': float(r1)/tot,
                'recall@5': float(r5)/tot,
                'recall@10': float(r10)/tot,
                'avg_rank': float(rank_tot)/tot
              }
        return ret

    cap_retrieval_recall = get_recall_k(i2c, 1, img2texts)

    ret = OrderedDict()
    ret['cap_avg_rank'] = cap_retrieval_recall['avg_rank']
    ret['cap_recall@1'] = cap_retrieval_recall['recall@1']
    ret['cap_recall@5'] = cap_retrieval_recall['recall@5']
    ret['cap_recall@10'] = cap_retrieval_recall['recall@10']
    ret['avg_cap_recall'] = (cap_retrieval_recall['recall@1'] + cap_retrieval_recall['recall@5'] + cap_retrieval_recall['recall@10']) /3
    ret['key_eval'] = "cap_recall@1"
    return ret


def rank_acc(score_matrix, tot_sample, candidate_len):
    """
    shape is (tot_sample * candidate_len, 4)
    score, ids, real_label, fake_label
    """
    assert score_matrix.shape[1] == 4
    # rank by ids
    score_matrix = np.take(score_matrix, np.argsort(score_matrix[:, 1]), axis=0)
    score_matrix = np.reshape(score_matrix, [tot_sample, candidate_len, 4])
    acc = 0
    for i in range(tot_sample):
        score_mat = score_matrix[i]
        top_idx = np.argsort(score_mat[:, 0])[::-1][0]
        acc = acc + int(score_mat[top_idx][2] == score_mat[top_idx][3])
    ret = OrderedDict()
    ret['rank_acc'] = round(float(acc/tot_sample), 4)
    ret['key_eval'] = "rank_acc"
    return ret

def test_recall_at_k():
    for i in range(3):
        for img_len in [3, 4, 5]:
            cap_ids = list(range(img_len * 5))
            text_len = len(cap_ids)
            img2texts, text2img = {}, {}
            for img in range(img_len):
                texts =  cap_ids[img*5 : (img+1)*5]
                img2texts[img] =  texts
                for text in texts:
                    text2img[text] = img
            score_mat = np.random.rand(img_len * img_len * 5, 1)
            imgs, texts = list(img2texts.keys()), list(text2img.keys())
            img_text = []
            for i in range(img_len):
                for j in range(text_len):
                    mat = [imgs[i], texts[j]]
                    img_text.append(mat)
            img_text = np.array(img_text)
            score_mat = np.concatenate([score_mat, img_text], axis=1)

            #print(img2texts)
            #print(text2img)
            #print(score_mat)
            ret = recall_at_k(score_mat, text2img, img2texts)
            for k, v in ret.items():
                print(k, v)
            print('---------------')

def test_rank_acc():
    for k in range(10):
        tot_sample, candidate_len, acc = 100, 1000, 0
        score_matrix = None
        real_label = list(range(candidate_len))
        for i in range(tot_sample):
            score_mat = np.random.random((candidate_len, 4))
            fake_label = np.array([np.random.randint(2) for i in range(candidate_len)]) * real_label
            score_mat[:, 1] = i
            score_mat[:, 2] = real_label
            score_mat[:, 3] = fake_label
            score_matrix = score_mat if score_matrix is None else np.concatenate([score_matrix, score_mat], axis=0)
            top_idx = np.argsort(score_mat[:, 0])[::-1][0]
            acc = acc + int(score_mat[top_idx][2] == score_mat[top_idx][3])
        np.random.shuffle(score_matrix)
        # print(score_matrix)
        ret = rank_acc(score_matrix, tot_sample, candidate_len)
        cul_acc = ret[ret['key_eval']]
        acc = round(float(acc/tot_sample), 4)
        assert cul_acc == acc
        print(cul_acc, acc)


if __name__ == "__main__":
    # test_recall_at_k()
    test_rank_acc()

