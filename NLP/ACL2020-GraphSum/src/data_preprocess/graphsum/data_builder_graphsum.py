#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import glob
import json
import os
import math
from collections import namedtuple
from os.path import join as pjoin
import codecs
from multiprocessing import Pool
import multiprocessing.pool
import numpy as np
from collections import defaultdict

import sentencepiece
from gensim import corpora, models, similarities
# from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize


class NoDaemonProcess(multiprocessing.Process):
    """ make 'daemon' attribute always return False"""
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NoDaemonProcessPool(multiprocessing.pool.Pool):
    """We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    because the latter is only a wrapper function, not a proper class."""
    Process = NoDaemonProcess


def format_to_json(args):
    """Constuct json format dataset"""
    corpora_files = {'train': [args.train_src, args.train_tgt],
                     'valid': [args.valid_src, args.valid_tgt],
                     'test': [args.test_src, args.test_tgt]}

    for corpus_type in ['train', 'valid', 'test']:
        src_files = codecs.open(corpora_files[corpus_type][0], 'r', 'utf-8').readlines()
        tgt_files = codecs.open(corpora_files[corpus_type][1], 'r', 'utf-8').readlines()
        assert len(src_files) == len(tgt_files), "the number of src files is not equal with tgt files!"

        a_lst = [(src, tgt, args) for src, tgt in zip(src_files, tgt_files)]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_json, a_lst):
            dataset.append(d)
            if len(dataset) > args.shard_size:
                pt_file = pjoin(args.json_path, "{:s}.{:d}.json".format(corpus_type, p_ct))
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset, ensure_ascii=False))
                    print("Written {:s}.{:d}.json".format(corpus_type, p_ct))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()

        if len(dataset) > 0:
            pt_file = pjoin(args.json_path, "{:s}.{:d}.json".format(corpus_type, p_ct))
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset, ensure_ascii=False))
                print("Written {:s}.{:d}.json".format(corpus_type, p_ct))
                p_ct += 1
                dataset = []


def _format_to_json(params):
    """Constuct json format dataset"""
    src, tgt, args = params

    docs = []
    # source input
    docs_strs = src.split('story_separator_special_tag')

    for doc_str in docs_strs:
        # split into sentences
        doc_lines = doc_str.strip().split('     ')
        # split each sentence into tokens and remove empty sentence
        doc_sents = [sent.strip().replace(r' +', ' ').split() for sent in doc_lines if sent.strip() != '']
        docs.append(doc_sents)

    # target summary
    tgt = tgt[1:].strip()
    return {'src': docs, 'tgt': tgt}


def format_to_paddle(args):
    """Constuct dataset for paddle models"""

    if args.dataset != '':
        datasets = [args.dataset]
    else:
        datasets = ['train', 'valid', 'test']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.json_path, '*' + corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((json_f, args, pjoin(args.data_path,
                                              "MultiNews." + str(args.max_nsents)
                                              + "." + real_name)))

        pool = NoDaemonProcessPool(args.n_cpus)
        for d in pool.imap_unordered(_format_to_paddle, a_lst):
            pass
        pool.close()
        pool.join()


def _format_to_paddle(params):
    """Constuct dataset for paddle models"""

    json_file, args, save_file = params
    if os.path.exists(save_file):
        print('Ignore %s' % save_file)
        return

    summ_data = SummData(args)

    print('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    total_sens = 0.
    total_words = 0.
    total_docs = 0.
    total_larger = 0.
    total_sum = 0.
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        b_data = summ_data.preprocess(source, tgt)
        if b_data is None:
            continue

        src_token_ids, tgt_token_ids, src_tokens, src_txt, tgt_txt, sen_num, word_num = \
            b_data.src_token_ids, b_data.tgt_token_ids, b_data.src_filtered, \
            b_data.original_src_txt, b_data.tgt_txt, b_data.src_filtered_len, \
            b_data.total_words_short

        total_sens += sen_num
        total_words += word_num
        total_docs += 1

        if args.sim_function == "tf-idf":
            sim_graph, larger_s, sum_s = construct_tfidf_sim_graph_by_gensim(src_tokens, args)
        elif args.sim_function == "lsi":
            sim_graph, larger_s, sum_s = construct_lsi_sim_graph(src_tokens, args)
        elif args.sim_function == "lda":
            res = construct_lda_sim_graph(src_tokens, args)
            if res is None:
                continue
            else:
                sim_graph, larger_s, sum_s = res
        else:
            raise ValueError("The sim function has been set wrong!")

        total_larger += larger_s
        total_sum += sum_s

        sim_graph = sim_graph
        b_data_dict = {"src": src_token_ids, "tgt": tgt_token_ids,
                       'src_str': src_txt, "tgt_str": tgt_txt,
                       'sim_graph': sim_graph}
        datasets.append(b_data_dict)

    print(len(datasets))
    print('Saving to %s' % save_file)
    print('total_docs:%s    total_sens:%s    toatl_words:%s' % (total_docs, total_sens, total_words))
    print('#sen/doc:%s    #word/doc:%s    #word/sen:%s' % (
    total_sens / total_docs, total_words / total_docs, total_words / total_sens))
    print('The ratio of similarity larger than %s is %s' % (args.sim_threshold, total_larger / (total_sum + 1e-18)))
    with open(save_file, 'w') as save:
        save.write(json.dumps(datasets, ensure_ascii=False))
    datasets = []
    gc.collect()


def construct_tfidf_sim_graph(sents, args):
    """Constuct TFIDF similarity graph"""

    sim_graph = []
    bows = []
    doc_freq = defaultdict(float)
    for sent in sents:
        term_freq = defaultdict(float)
        for token in sent:
            term_freq[token] += 1.0
        bows.append(term_freq)

        for term in term_freq.keys():
            doc_freq[term] += 1.0

    for i in range(len(sents)):
        bow_i = bows[i]
        for term in bow_i:
            bow_i[term] = bow_i[term] * math.log(float(len(sents)) / (doc_freq[term] + 1e-18))

    sim_memory = defaultdict(float)
    total = 0.
    count_large = 0.
    for i in range(len(sents)):
        sim = []
        for j in range(len(sents)):
            key1 = str(i) + ":" + str(j)
            key2 = str(j) + ":" + str(i)
            if i == j:
                sim.append(1.0)
            elif key1 in sim_memory:
                sim.append(sim_memory.get(key1))
            elif key2 in sim_memory:
                sim.append(sim_memory.get(key2))
            else:
                sim_ij = compute_tfidf_sim(bows[i], bows[j], doc_freq.keys())
                sim_memory[key1] = sim_ij
                sim_memory[key2] = sim_ij
                sim.append(sim_ij)

                total += 1
                if sim_ij > args.sim_threshold:
                    count_large += 1

        sim_graph.append(sim)

    # print("Large than 0.02: %s" % (count_large / total))

    return sim_graph, count_large, total


def compute_tfidf_sim(bow1, bow2, word_dict):
    """compute tfidf similarity"""
    sum_all, sum1, sum2 = 0.0, 0.0, 0.0
    for term in word_dict:
        sum_all += bow1[term] * bow2[term]
        sum1 += bow1[term] * bow1[term]
        sum2 += bow2[term] * bow2[term]

    sim = sum_all / (math.sqrt(sum1) * math.sqrt(sum2) + 1e-18)
    return sim


def construct_tfidf_sim_graph_by_gensim(corpus, args):
    """Constuct TFIDF similarity graph by Gensim package"""

    sim_graph = []
    raw_corpus = [' '.join(para) for para in corpus]

    # create English stop words list
    stoplist = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in para.lower().split() if word not in stoplist]
             for para in raw_corpus]
    # Create a set of frequent words
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # stem each word
    processed_corpus = [[p_stemmer.stem(token) for token in text if frequency[token] > 1] for text in texts]

    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # train the model
    tfidf = models.TfidfModel(bow_corpus)
    # transform the "system minors" string
    corpus_tfidf = tfidf[bow_corpus]

    for i, cor in enumerate(corpus_tfidf):
        if len(cor) == 0:
            print("The corpus_tfidf[i] is None: %s" % str(corpus_tfidf[i]))
            print(bow_corpus[i])
            # exit(1)

    index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

    total = 0.
    count_large = 0.
    for i in range(len(corpus_tfidf)):
        sim = index[corpus_tfidf[i]]
        assert len(sim) == len(corpus_tfidf), "the tfidf sim is not correct!"
        sim_graph.append(sim.tolist())

        for s in sim:
            total += 1
            if s > args.sim_threshold:
                count_large += 1

    print("sim_graph[0]: %s" % str(sim_graph[0]))

    return sim_graph, count_large, total


def get_optimal_lsimodel_by_coherence_values(corpus, texts, dictionary, stop=100, start=10, step=10):
    """
    get the lsi model with optimal number of topics

    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    num_lists = range(start, stop, step)
    for num_topics in num_lists:
        # generate LSA model
        model = models.LsiModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)  # train model
        model_list.append(model)
        coherencemodel = models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    print("num_topics: %s" % str(num_lists))
    print("coherence_values: %s" % str(coherence_values))

    max_ind = np.argmax(np.array(coherence_values))
    print("opt_num_topics: %s" % num_lists[max_ind])
    return model_list[max_ind]


def construct_lsi_sim_graph(corpus, args):
    """
    compute lsi vector similarity between paragraphs
    :param corpus:
    :param args:
    :return:
    """
    sim_graph = []
    raw_corpus = [' '.join(para) for para in corpus]

    # create English stop words list
    stoplist = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in para.lower().split() if word not in stoplist]
             for para in raw_corpus]
    # Create a set of frequent words
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # stem each word
    processed_corpus = [[p_stemmer.stem(token) for token in text] for text in texts]

    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # train the model
    tfidf = models.TfidfModel(bow_corpus)
    # transform the "system minors" string
    corpus_tfidf = tfidf[bow_corpus]

    if args.find_opt_num:
        lsi = get_optimal_lsimodel_by_coherence_values(corpus=corpus_tfidf, texts=processed_corpus,
                                                       dictionary=dictionary)
    else:
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary,
                              num_topics=args.num_topics)  # initialize an LSI transformation

    corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

    # for i, doc in enumerate(corpus_lsi):
    #     if len(doc) == 0:
    #         print("The lsi is empty: %s" % raw_corpus[i])

    index = similarities.MatrixSimilarity(corpus_lsi, num_features=len(dictionary))

    total = 0.
    count_large = 0.
    for i in range(len(corpus_lsi)):
        sim = index[corpus_lsi[i]]

        assert len(sim) == len(corpus_lsi), "the lsi sim is not correct!"
        sim_graph.append(sim)

        for s in sim:
            total += 1
            if s > args.sim_threshold:
                count_large += 1

    print("sim_graph[0]: %s" % str(sim_graph[0]))
    return sim_graph, count_large, total


def get_optimal_ldamodel_by_coherence_values(corpus, texts, dictionary, stop=100, start=10, step=10):
    """
    get the lsi model with optimal number of topics

    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LDA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    num_lists = range(start, stop, step)
    for num_topics in num_lists:
        # generate LDA model
        model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                                alpha='auto', eta='auto', eval_every=None)  # train model
        model_list.append(model)
        coherencemodel = models.CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    print("num_topics: %s" % str(num_lists))
    print("coherence_values: %s" % str(coherence_values))

    max_ind = np.argmax(np.array(coherence_values))
    print("opt_num_topics: %s" % num_lists[max_ind])
    return model_list[max_ind]


def construct_lda_sim_graph(corpus, args):
    """
    compute lda vector similarity between paragraphs
    :param corpus:
    :param args:
    :return:
    """
    sim_graph = []
    raw_corpus = [' '.join(para) for para in corpus]

    # create English stop words list
    stoplist = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in para.lower().split() if word not in stoplist]
             for para in raw_corpus]
    # Create a set of frequent words
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # stem each word
    processed_corpus = [[p_stemmer.stem(token) for token in text] for text in texts]

    dictionary = corpora.Dictionary(processed_corpus)

    if len(dictionary) < args.num_topics:
        return None

    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    # train the model
    if args.find_opt_num:
        lda = get_optimal_ldamodel_by_coherence_values(corpus=bow_corpus, texts=processed_corpus, dictionary=dictionary)
    else:
        lda = models.LdaModel(corpus=bow_corpus, num_topics=args.num_topics, id2word=dictionary,
                              alpha='auto', eta='auto', eval_every=None, minimum_probability=0.0)
    # LdaMulticore(bow_corpus, id2word=dictionary, num_topics=args.num_topics, eta='auto',
    # eval_every=None, minimum_probability=0.0)

    corpus_lda = lda[bow_corpus]  # create a double wrapper over the original corpus: bow->lda
    index = similarities.MatrixSimilarity(corpus_lda, num_features=len(dictionary))

    print("corpus_lda[0]: %s" % str(corpus_lda[0]))

    total = 0.
    count_large = 0.
    for i in range(len(corpus_lda)):
        sim = index[corpus_lda[i]]

        assert len(sim) == len(corpus_lda), "the lda sim is not correct!"
        sim_graph.append(sim)

        for s in sim:
            total += 1
            if s > args.sim_threshold:
                count_large += 1

    print("sim_graph[0]: %s" % str(sim_graph[0]))
    return sim_graph, count_large, total


class SummData(object):
    """Process data by sentencepiece tokenization"""
    def __init__(self, args):
        self.args = args
        """load vocabulary"""
        self.spm = sentencepiece.SentencePieceProcessor()
        self.spm.load(args.vocab_path)
        self.symbols = {'BOS': self.spm.PieceToId('<S>'), 'EOS': self.spm.PieceToId('</S>'),
                        'PAD': self.spm.PieceToId('<PAD>'), 'EOT': self.spm.PieceToId('<T>'),
                        'EOP': self.spm.PieceToId('<P>'), 'EOQ': self.spm.PieceToId('<Q>')}
        self.vocab_size = len(self.spm)

    def preprocess(self, src, tgt):
        """preprocess src and tgt by tokenization"""
        if len(src) == 0:
            return None

        docs = []
        for doc in src:
            # remove too short sentence
            doc_filtered = [sent for sent in doc if len(sent) > self.args.min_src_ntokens]
            if len(doc_filtered) > 0:
                docs.append(doc_filtered)

        docs_num = len(docs)
        total_sens = sum([len(doc) for doc in docs])
        total_words = 0
        for doc in docs:
            for sent in doc:
                total_words += len(sent)

        print("doc_num: %s    total_sens:%s    total_words:%s" % (docs_num, total_sens, total_words))

        # truncate long documents, combine lead sentences of each document
        if total_sens > self.args.max_nsents:
            docs_filtered = _filter_and_combine_docs(docs, list(range(docs_num)), self.args.max_nsents, docs_num)
        else:
            docs_filtered = docs

        src_filtered = []
        for doc in docs_filtered:
            src_filtered.extend(doc)

        total_words = sum([len(sent) for sent in src_filtered])
        print("After filtered, total_sens:%s    total_words:%s" % (str(len(src_filtered)), total_words))

        assert len(src_filtered) <= self.args.max_nsents, "docs filtered wrong!!!"

        # discard too short documents
        if len(src_filtered) < self.args.min_nsents:
            return None

        # truncate too long sentence
        src_filtered_short = [sent[:self.args.max_src_ntokens] for sent in src_filtered]
        src_txt = [' '.join(sent) for sent in src_filtered_short]
        original_src_txt = [' '.join(sent) for sent in src_filtered]

        src_token_ids = [self.spm.encode_as_ids(sent) for sent in src_txt]

        tgt_sents = sent_tokenize(tgt)
        tgt_sents_ids = [self.spm.encode_as_ids(sent) for sent in tgt_sents]
        tgt_token_ids = [self.symbols['BOS']] + sum([p + [self.symbols['EOQ']] for p in tgt_sents_ids], [])[:-1] + \
                        [self.symbols['EOS']]  # <S> + ...<Q> + ...<Q> + ...</S>

        tgt_txt = tgt
        total_words_short = sum([len(sent) for sent in src_filtered_short])

        res = namedtuple('result', ['src_token_ids', 'tgt_token_ids', 'src_filtered',
                                    'original_src_txt', 'tgt_txt', 'src_filtered_len',
                                    'total_words_short'])

        return res(src_token_ids=src_token_ids, tgt_token_ids=tgt_token_ids,
                   src_filtered=src_filtered, original_src_txt=original_src_txt,
                   tgt_txt=tgt_txt, src_filtered_len=len(src_filtered_short),
                   total_words_short=total_words_short)


def _filter_and_combine_docs(docs, cur_doc_ids, total_sents, cur_num):
    """Combine lead num of tokens from each document to get multi-document input"""
    avg = total_sents // cur_num
    exceed_doc_ids = [i for i in cur_doc_ids if len(docs[i]) > avg]
    if len(exceed_doc_ids) == cur_num:
        return [doc[:avg] for doc in docs]

    for i in cur_doc_ids:
        if i not in exceed_doc_ids:
            total_sents -= len(docs[i])

    return _filter_and_combine_docs(docs, exceed_doc_ids, total_sents, len(exceed_doc_ids))
