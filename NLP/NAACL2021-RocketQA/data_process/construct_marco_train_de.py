import sys
import random
sys.path.append('data_process/')
import utils

recall_cands_file = sys.argv[1]
ce_score_file = sys.argv[2]
outfile = sys.argv[3]

random_seed = 111
rng = random.Random(random_seed)

neg_cnt = 4
ce_threshold_neg = 0.1
ce_threshold_pos = 0.9

q_text, p_text, p_title = utils.load_corpus(corpus='marco', q_type='train')
pos_qp, pos_qp_add = utils.load_pos_examples(p_text)
cand_qp_all, train_qids = utils.load_candidates(recall_cands_file, col=4)
ce_score = utils.load_ce_score(ce_score_file, train_qids)

# neg examples
neg_qp = {}
for qid, pids in cand_qp_all.items():
    if qid not in pos_qp:
        continue
    select_pid = []
    pos_cnt = len(pos_qp[qid])
    for index in range(50):
        _pid = pids[index]
        if len(select_pid) == neg_cnt * pos_cnt:
            break 
        if _pid in pos_qp[qid] or _pid in select_pid or _pid in pos_qp_add.get(qid, []):
            continue
        if ce_score[qid][index] < ce_threshold_neg:
            select_pid.append(_pid)

    while len(select_pid) < neg_cnt * pos_cnt:
        _pid = rng.choice(pids[50:])
        if _pid in pos_qp[qid] or _pid in select_pid or _pid in pos_qp_add.get(qid, []):
            continue
        select_pid.append(_pid)

    neg_qp[qid] = select_pid

with open(outfile, 'w') as out:
    for qid in pos_qp:
        neg_pids = neg_qp[qid]
        for i in range(neg_cnt):
            for pos_pid in pos_qp[qid]:
                neg_pid = neg_pids.pop()
                out.write('%s\t%s\t%s\t%s\t%s\t0\n' % (q_text[qid], 
                    p_title.get(pos_pid, '-'), p_text[pos_pid], 
                    p_title.get(neg_pid, '-'), p_text[neg_pid]))
