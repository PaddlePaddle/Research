import argparse
import os
import time
from multiprocessing import Pool
import tools.my_pyrouge as pyrouge
import shutil
import sys
import codecs


def process(data):
    """process the data to build files for ROUGE evaluation"""
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = "rouge-tmp-{}-{}".format(current_time, pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")

    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])

        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)

    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def test_rouge(cand, ref, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    print('Num of candidates: %d' % len(candidates))
    print('Num of references: %d' % len(references))
    assert len(candidates) == len(references), "!!!!!!! Note: The number of candidates is " \
                                               "not equal to the number of references!!!!!!!"

    candidates_chunks = list(chunks(candidates, int(len(candidates)/num_processes)))
    references_chunks = list(chunks(references, int(len(references)/num_processes)))

    n_pool = len(candidates_chunks)
    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i], references_chunks[i], i))

    pool = Pool(n_pool)
    results = pool.imap(process, arg_lst)
    pool.close()
    pool.join()

    final_results = {}
    for i, r in enumerate(results):
        for k in r:
            if k not in final_results:
                final_results[k] = r[k]*len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k]/len(candidates)

    return final_results


def rouge_results_to_str(results_dict):
    """report rouge results"""
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_recall"] * 100
        # ,results_dict["rouge_su*_f_score"] * 100
    )


if __name__ == "__main__":
    # init_logger('test_rouge.log')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='candidate file')
    parser.add_argument('-r', type=str, default="reference.txt",
                        help='reference file')
    parser.add_argument('-p', type=int, default=1,
                        help='number of processes')
    args = parser.parse_args()
    print(args.c)
    print(args.r)
    print(args.p)
    if args.c.upper() == "STDIN":
        candidates = sys.stdin
    else:
        candidates = codecs.open(args.c, encoding="utf-8")
    references = codecs.open(args.r, encoding="utf-8")

    results_dict = test_rouge(candidates, references, args.p)
    # return 0
    print(time.strftime('%H:%M:%S', time.localtime()))
    print(rouge_results_to_str(results_dict))
    # logger.info(rouge_results_to_str(results_dict))
