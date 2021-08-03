import sys
import numpy as np
sys.path.append('data_process/')
sys.path.append('metric/')
from dpr.utils.tokenizers import SimpleTokenizer
import utils
import unicodedata

recall_cands_file = sys.argv[1]
topk = 100

answers = utils.load_answers('test')
q_text, p_text, p_title = utils.load_corpus(corpus='nq', q_type='test')
cand_qp_all, train_qids = utils.load_candidates(recall_cands_file, col=4)

def has_answer(answers, text, tokenizer, match_type):
    text = unicodedata.normalize('NFD', text)
    if match_type == 'string':
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answers:
            single_answer = unicodedata.normalize('NFD', single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i+ len(single_answer)]:
                    return 1
    return 0

print('calculating acc')
right_top100 = set()
right_top50 = set()
right_top20 = set()
right_top10 = set()
right_top5 = set()
tok_opts = {}
tokenizer = SimpleTokenizer(**tok_opts)
for qid, pids in cand_qp_all.items():
    answer = answers[qid]
    for i, pid in enumerate(pids):
        if has_answer(answer, p_text[pid], tokenizer, 'string'):
            if i < 100:
                right_top100.add(qid)
                if i < 50:
                    right_top50.add(qid)
                    if i < 20:
                        right_top20.add(qid)
                        if i < 10:
                            right_top10.add(qid)
                            if i < 5:
                                right_top5.add(qid)
                                break

query_num = len(cand_qp_all)
print(query_num)
print(len(right_top100))
r100 = len(right_top100) * 1.0 / query_num
r50 = len(right_top50) * 1.0 / query_num
r20 = len(right_top20) * 1.0 / query_num
r10 = len(right_top10) * 1.0 / query_num
r5 = len(right_top5) * 1.0 / query_num

print('recall@100: ' +  str(r100))
print('recall@50: ' + str(r50))
print('recall@20: ' + str(r20))
print('recall@10: ' + str(r10))
print('recall@5: ' + str(r5))

