""" generate source and target data for sub-question generation """

import json

folder = "./"
fns = ['train_clause_data.json', 'train_group.json']
sents_src = []
sents_tgt = []
for fn in fns:
    fn = folder + fn
    f = open(fn, 'r')
    for line in f:
        data = json.loads(line.strip())
        if not data['sub_sql_new']:
            # remove multi table data
            if 't1' in data['sub_sql'] or 't2' in data['sub_sql'] or 't3' in data['sub_sql']:
                print('skip ' + data['sub_sql'])
                continue
            
            # replace OP
            # for op, nl in zip(WHERE_OPS, WHERE_OPS_NL):
            # op_str = f" {op} "
            # if  op_str in sub_sql:
            #     new_sub_sql = new_sub_sql.replace(op_str, f" {nl} ")
            # one_data['sun_sql_new'] = new_sub_sql
            
            data['sub_sql_new'] = data['sub_sql']
        if data['sub_sql_new']:
            sub_query = data["db_id"] + ' | ' + data['sub_sql_new']
            sub_question = data["sub_ques"]
            
            sents_src.append(sub_query.replace('_', ' '))
            sents_tgt.append(sub_question)
        

idx = int(len(sents_src) * 4 / 5)
train_src = sents_src[:idx]
train_tgt = sents_tgt[:idx]

dev_src = sents_src[idx:]
dev_tgt = sents_tgt[idx:]

with open(folder + 'train_src.txt', 'w')as fs, open(folder + 'train_tgt.txt', 'w')as ft:
    cache = {}
    for src, tgt in zip(train_src, train_tgt):
        if not src:
            continue
        if src not in cache:
            cache[src] = 1
            fs.write(src + ' # 1 \n')
        else:
            cache[src] = cache[src] + 1
            fs.write(src + ' # %d \n' % cache[src])
        ft.write(tgt + '\n')

with open(folder + 'dev_src.txt', 'w')as fs, open(folder + 'dev_tgt.txt', 'w')as ft:
    cache = {}
    for src, tgt in zip(dev_src, dev_tgt):
        if not src:
            continue
        if src not in cache:
            cache[src] = 1
            fs.write(src + ' # 1 \n')
        else:
            cache[src] = cache[src] + 1
            fs.write(src + ' # %d \n' % cache[src])
        ft.write(tgt + '\n')
