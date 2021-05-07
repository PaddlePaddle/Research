import sys
import random
sys.path.append('data_process/')
import utils

recall_cands_file = sys.argv[1]
ce_score_file = sys.argv[2]
outfile = sys.argv[3]
corpus = sys.argv[4]

random_seed = 111
rng = random.Random(random_seed)

neg_cnt = 1
if corpus == 'marco':
    neg_cnt = 4
ce_threshold_neg = 0.1
ce_threshold_pos = 0.9

q_text, p_text, p_title = utils.load_corpus(corpus=corpus, q_type='train', unlabel=True)
cand_qp_all, train_qids = utils.load_candidates(recall_cands_file, col=4)
ce_score = utils.load_ce_score(ce_score_file, train_qids)

# pseudo pos examples
pos_qp = {}
for qid, pids in cand_qp_all.items():
    select_pid = [ce_threshold_pos]
    for index in range(50):
        _pid = pids[index]
        score = ce_score[qid][index]
        if score > select_pid[-1]:
            select_pid = [_pid, score]
    if select_pid[-1] > ce_threshold_pos:
        pos_qp[qid] = select_pid[0]

# neg examples
neg_qp = {}
for qid, pids in cand_qp_all.items():
    if qid not in pos_qp:
        continue
    select_pid = []
    pos_cnt = 1
    for index in range(50):
        _pid = pids[index]
        if len(select_pid) == neg_cnt * pos_cnt:
            break 
        if _pid in pos_qp[qid] or _pid in select_pid:
            continue
        if ce_score[qid][index] < ce_threshold_neg:
            select_pid.append(_pid)

    neg_qp[qid] = select_pid

with open(outfile, 'w') as out:
    for qid, neg_pids in neg_qp.items():
        pos_pid = pos_qp[qid]
        for neg_pid in neg_pids:
            out.write('%s\t%s\t%s\t%s\t%s\t0\n' % (q_text[qid], 
                    p_title.get(pos_pid, '-'), p_text[pos_pid], 
                    p_title.get(neg_pid, '-'), p_text[neg_pid]))
