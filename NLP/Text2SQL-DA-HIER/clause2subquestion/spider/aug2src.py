""" trans demo aug sql to question """
import json

fn_in = "clause_aug_sample.json"
fn_src = "aug_src.txt"

with  open(fn_in)as f, open(fn_src, 'w')as f1:
    
    srcs = {}
    
    for line in f:
        one_data = json.loads(line.strip())
        sub_sql_list = one_data['sub_sql_list']
        if any([len(i) == 0 for i in sub_sql_list]):
            continue
        db_id = one_data['db_id']
        db_id = db_id.replace("_", " ")
        for sub_sql in sub_sql_list:
            if sub_sql in srcs:
                srcs[sub_sql] = srcs[sub_sql] + 1
                f1.write(db_id + ' | ' + sub_sql.strip() + ' # %d\n' % srcs[sub_sql])
            else:
                srcs[sub_sql] = 1
                f1.write(db_id + ' | ' + sub_sql.strip() + ' # 1\n')
        