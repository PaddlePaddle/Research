import sys
import os

cur_path = os.path.dirname(os.path.realpath(__file__))
corpus_path = 'corpus/'

def load_id_text(file_name):
    """load tsv files"""
    id_text = {}
    with open(file_name) as inp:
        for line in inp:
            line = line.strip()
            id, text = line.split('\t')
            id_text[id] = text
    return id_text

def load_corpus(corpus='marco', q_type='train', unlabel=False):
    """load corpus"""
    if not unlabel:
        q_file = os.path.join(corpus_path, corpus, '%s.query.txt' % q_type)
    elif corpus == 'marco':
        q_file = os.path.join(corpus_path, 'augment/orcas_yahoo_nq.query.txt')
    else:
        q_file = os.path.join(corpus_path, 'augment/mrqa.query.txt')
    q_text = load_id_text(q_file)

    p_file = os.path.join(corpus_path, corpus, 'para.txt')
    p_text = load_id_text(p_file)

    t_file = os.path.join(corpus_path, corpus, 'para.title.txt')
    p_title = load_id_text(t_file)

    print('load all corpus done!')
    return q_text, p_text, p_title

def load_answers(q_type='train'):
    """load exist answers in NQ"""
    qid_answers = {}
    file = os.path.join(corpus_path, 'nq/%s.answers.txt' % q_type)
    with open(file) as inp:
        for line in inp:
            info = line.strip().split('\t')
            qid = info[0]
            answers = info[1:]
            qid_answers[qid] = []
            for ans in answers:
                ans = ans.strip('.').lower()
                qid_answers[qid].append(ans)
    print('has answer qids: %s' % len(qid_answers))
    return qid_answers

def has_answer(text, answers):
    for answer in answers:
        text = text.strip().lower().replace(' ', '')
        answer = answer.strip().lower().replace(' ', '')
        if text.find(answer) != -1:
            return True
    return False

def load_pos_examples(p_text):
    """positive examples(only for MSMARCO)"""
    pos_qp = {}
    file = os.path.join(corpus_path, 'marco/qrels.train.tsv')
    with open(file) as inp:
        for line in inp:
            line = line.strip()
            qid, pid = line.split('\t')
            if qid not in pos_qp:
                pos_qp[qid] = []
            pos_qp[qid].append(pid)
    print('positive qids: %s' % len(pos_qp))

    # additional positive examples(collect by literal match)
    pos_qp_add = {}
    file_add = os.path.join(corpus_path, 'marco/qrels.train.addition.tsv')
    with open(file_add) as inp:
        for line in inp:
            qid, pid = line.strip().split('\t')
            if qid not in pos_qp_add:
                pos_qp_add[qid] = []
            pos_qp_add[qid].append(pid)
    return pos_qp, pos_qp_add

def load_candidates(filename, col=4, topk=0):
    """Top K candidate examples"""
    cand_qp_all = {}
    train_qids = []
    with open(filename) as inp:
        for line in inp:
            line = line.strip()
            if col == 4:
                qid, pid, idx, score = line.split('\t')
            else:
                qid, pid, idx = line.split('\t')
            if topk > 0 and int(idx) > topk:
                continue
            if qid not in cand_qp_all:
                cand_qp_all[qid] = []
                train_qids.append(qid)
            cand_qp_all[qid].append(pid)
    print('load candidate qids: %s' % len(cand_qp_all))
    return cand_qp_all, train_qids

def load_ce_score(filename, train_qids, topk=50):
    """Top K cross_encoder model score"""
    ce_score = {}
    with open(filename) as inp:
        for i, line in enumerate(inp):
            line = line.strip()
            score = float(line)
            qid = train_qids[i//topk]
            if qid not in ce_score:
                ce_score[qid] = []
            ce_score[qid].append(score)
    print('load cross_encoder score: %s' % len(ce_score))
    return ce_score

if __name__ == '__main__':
    load_answers()
