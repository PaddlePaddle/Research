import sys
import random
import utils

recall_cands_file = sys.argv[1]
ce_score_file = sys.argv[2]
outfile = sys.argv[3]

neg_cnt = 4
ce_threshold_neg = 0.1
ce_threshold_pos = 0.9

q_text, p_text, p_title = utils.load_corpus(corpus='nq', q_type='train')
answers = utils.load_answers(q_type='train')
cand_qp_all, train_qids = utils.load_candidates(recall_cands_file, col=4)
ce_score = utils.load_ce_score(ce_score_file, train_qids, topk=100)

out = open(outfile, 'w')
for qid, pids in cand_qp_all.items():
    pos_pid = ''
    neg_pid = ''
    for index in range(100):
        _pid = pids[index]
        if utils.has_answer(p_text[_pid], answers[qid]) or utils.has_answer(p_title[_pid], answers[qid]):
            if not pos_pid:
                pos_pid = _pid
        else:
            if not neg_pid and ce_score[qid][index] < ce_threshold_neg:
                neg_pid = _pid
        if pos_pid and neg_pid:
            out.write('%s\t%s\t%s\t%s\t%s\t0\n' % (q_text[qid],
                    p_title.get(pos_pid, '-'), p_text[pos_pid],
                    p_title.get(neg_pid, '-'), p_text[neg_pid]))
            break
out.close()
