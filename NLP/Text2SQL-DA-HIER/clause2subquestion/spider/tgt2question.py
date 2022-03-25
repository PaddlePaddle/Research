"""
trans target sub-question to full question
"""

import json
import random

def combine_question(sub_questions, sub_sqls):
    """sub-questuiion 2 full question

    Args:
        sub_questions ([type]): sub questions
        sub_sqls ([type]): sub sqls

    Returns:
        string: full question
    """
    where_dict = {4:'', 1:'where ', 2:'for ', 3:'when ', 0:', '}
    order_dict = {2:'', 1:' with ', 0:', ', 3:' in '}
    select_dict = {0:', ', 1:'give '}
    group_dict = {2:'', 1:' for ', 0:', '}
    base_dict = {0:', ', 1:''}
    len_sub = len(sub_questions)
    full_question = ''
    for sub_q, sub_sql in zip(sub_question, sub_sqls):
        sub_sql = sub_sql.split()
        if sub_sql[0].lower() == 'where':
            idx = random.randint(0, len(where_dict) - 1)
            # where clause
            conj = where_dict[idx]
            if not full_question and idx == 0:
                idx = 4
        elif sub_sql[0].lower() == "order":
            idx = random.randint(0, len(order_dict)-1)
            # order clause
            conj = order_dict[idx]
            if not full_question and idx == 0:
                idx = 2
            pass
        elif sub_sql[0].lower() == 'select':
            idx = random.randint(0, len(select_dict)-1)
            conj = base_dict[idx]
            pass
        elif sub_sql[0].lower() == 'group':
            idx = random.randint(0, len(group_dict)-1)
            conj = group_dict[idx]
            pass
        else:
            idx = random.randint(0, len(base_dict)-1)
            conj = base_dict[idx]
        if conj == ',':
            sub_q = sub_q + ','
        elif conj != sub_q[0]:
            # avoid repeatition
            sub_q = conj + sub_q
        else:
            pass
        full_question = full_question + sub_q + ' '
    full_question = full_question.strip(',').strip(' ')
        
    return full_question       
        

if __name__ == "__main__":
    fn_in = "clause_aug_sample.json"
    fn_src = "aug_src.txt"
    fn_tgt = "aug_tgt.txt"
    fn_out = "aug_output.json"

    with open(fn_in)as f, open(fn_tgt, 'r')as f1, open(fn_out, 'w')as f2:
        subqs = [subq.strip() for subq in f1]
        idx = 0
        for line in f:
            one_data = json.loads(line.strip())
            sub_sql_list = one_data['sub_sql_list']
            sub_question = []
            for sub_sql in sub_sql_list:
                sub_question.append(subqs[idx])
                idx += 1
            if any([len(i) == 0 for i in sub_sql_list]):
                continue
            new_data = {}
            new_data['db_id'] = one_data['db_id']
            new_data['sql'] = one_data['sql']
            new_data['query'] = one_data['query']
            new_data['query_toks'] = one_data['query_toks']
            
            new_data['sub_questions'] = sub_question
            full_question = combine_question(sub_question, sub_sql_list)
            new_data['question'] = full_question
            
            f2.write(json.dumps(new_data, ensure_ascii=False) + '\n') 
                