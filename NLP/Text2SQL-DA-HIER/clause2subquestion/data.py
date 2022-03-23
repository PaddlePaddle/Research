# !/usr/bin/env python3
""" generate source and target data for sub-question generation """

import json

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
WHERE_OPS_NL = ('not', 'between', 'equal to', 'more than', 'less than', 'no less than', 'no more than', 'not equal to', 'in', 'like', 'is', 'exists')

def get_table_names(sent):
    tns = []
    for i in range(5):
        s = f"t{i}."
        if s in sent:
            tns.append(s)
    return tns


if __name__ == "__main__":
    folder = "spider/"
    sents_src = []
    sents_tgt = []
    fns = [folder + 'train_clause_data.json', folder + 'train_group.json']
    for fn in fns:
        with open(fn, 'r')as f:
            for line in f:
                data = json.loads(line.strip())
                if not data['sub_sql_new']:
                    # remove multi table data
                    sub_sql = data['sub_sql']
                    tns = get_table_names(data['sub_sql'])

                    # if 'sub_sql_new' not in data and len(tns) > 1:
                    #     print('skip ' + data['sub_sql'])
                    #     continue
                    
                    # replace OP
                    if 'sub_sql_new' not in data or not data['sub_sql_new']:
                        new_sub_sql = sub_sql
                        for op, nl in zip(WHERE_OPS, WHERE_OPS_NL):
                            op_str = f" {op} "
                            if  op_str in sub_sql:
                                new_sub_sql = new_sub_sql.replace(op_str, f" {nl} ")
                            data['sun_sql_new'] = new_sub_sql

                        # remove str like 't1'
                        for tn in tns:
                            data['sub_sql_new'] = data['sub_sql_new'].replace('tn', '')

                if data['sub_sql_new']:
                    sub_query = data["db_id"] + ' | ' + data['sub_sql_new']
                    sub_question = data["sub_ques"]
                    
                    sents_src.append(sub_query.replace('_', ' '))
                    sents_tgt.append(sub_question)

    # split train set and dev set
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
