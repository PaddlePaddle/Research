import sys
import random
sys.path.append('data_process/')
import utils

recall_cands_file = sys.argv[1]
outfile = sys.argv[2]

random_seed = 111
rng = random.Random(random_seed)

q_text, p_text, p_title = utils.load_corpus(corpus='nq', q_type='train')
answers = utils.load_answers(q_type='train')
cand_qp_all, train_qids = utils.load_candidates(recall_cands_file, col=4)

out = open(outfile, 'w')
for qid, pids in cand_qp_all.items():
    pos_pid = ''
    neg_pid_cand = []
    for index in range(100):
        _pid = pids[index]
        if utils.has_answer(p_text[_pid], answers[qid]) or utils.has_answer(p_title[_pid], answers[qid]):
            if not pos_pid:
                pos_pid = _pid
        else:
            neg_pid_cand.append(_pid)
    if pos_pid:
        out.write('%s\t%s\t%s\t1\n' % (q_text[qid], p_title.get(pos_pid, ''), p_text[pos_pid]))
    if neg_pid_cand:
        neg_pid = random.choice(neg_pid_cand)
        out.write('%s\t%s\t%s\t0\n' % (q_text[qid], p_title.get(neg_pid, ''), p_text[neg_pid]))
out.close()
