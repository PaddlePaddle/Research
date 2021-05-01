import sys
import random
sys.path.append('data_process/')
import utils

recall_cands_file = sys.argv[1]
outfile = sys.argv[2]

random_seed = 111
rng = random.Random(random_seed)

neg_cnt = 4

q_text, p_text, p_title = utils.load_corpus(corpus='marco', q_type='train')
pos_qp, pos_qp_new = utils.load_pos_examples(p_text)
cand_qp_all, train_qids = utils.load_candidates(recall_cands_file, col=4)

# neg examples
neg_qp = {}
for qid, pids in cand_qp_all.items():
    select_pid = []
    while len(select_pid) < neg_cnt:
        _pid = rng.choice(pids)
        if _pid in pos_qp[qid] or _pid in select_pid or _pid in pos_qp_new.get(qid, []):
            continue

        select_pid.append(_pid)
    neg_qp[qid] = select_pid

with open(outfile, 'w') as out:
    for qid in pos_qp:
        for pid in pos_qp[qid]:
            out.write('%s\t%s\t%s\t1\n' % (q_text[qid], p_title.get(pid, ''), p_text[pid]))
    for qid in neg_qp:
        for pid in neg_qp[qid]:
            out.write('%s\t%s\t%s\t0\n' % (q_text[qid], p_title.get(pid, ''), p_text[pid]))

