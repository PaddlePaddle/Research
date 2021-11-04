#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#

"""
File: data_builder_graphsum.py
Author: chenmoye(chenmoye@baidu.com)
Date: 2021-10-18 14:52
Desc:
"""

import re
import gc
import glob
import json
import os
import math
from collections import namedtuple
from os.path import join as pjoin
import codecs
# import torch
from multiprocessing import Pool
import multiprocessing.pool
import numpy as np
# from others.utils import clean
from collections import defaultdict
from utils import _get_word_ngrams
from roberta.roberta_tokenization import GptBpeTokenizer
import nltk
import itertools
import time

import sentencepiece
from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
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
    # for corpus_type in ["test"]:
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
    docs_strs = src.split('|||||')
    for doc_str in docs_strs:
        doc_lines = doc_str.strip()
        doc_lines = sent_tokenize(doc_str.strip())
        doc_sents = [sent.strip() for sent in doc_lines if sent.strip() != '']
        doc_sents = doc_sents[:args.max_nsents]
        docs.append(doc_sents)
    tgt = tgt.strip()
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
            a_lst.append((json_f, args, pjoin(args.data_path, real_name)))

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
        b_data = summ_data.preprocess(source, tgt, args)
        if b_data is None:
            continue

        src_token_ids, tgt_token_ids, src_tokens, tgt_txt, sent_labels, cls_ids, sep_ids, summary_rank = \
            b_data.src_token_ids, b_data.tgt_token_ids, b_data.src_txt, b_data.tgt_txt, \
            b_data.sent_labels, b_data.cls_ids, b_data.sep_ids, \
            b_data.summary_rank

        if len(src_token_ids) == 0:
            continue
    
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
        # print("sim_graph: " + str(sim_graph))
        b_data_dict = {"src": src_token_ids, "tgt": tgt_token_ids,
                       'src_str': src_tokens, "tgt_str": tgt_txt,
                       'sim_graph': sim_graph, "sent_labels": sent_labels, 
                       'cls_ids': cls_ids, 'sep_ids': sep_ids, 
                       "summary_rank": summary_rank}
        count_sent_num = 0
        for doc in src_tokens:
            count_sent_num += len(doc)
        count_graph_num = 0
        for doc in sim_graph:
            count_graph_num += len(doc)
            for graph in doc:
                assert count_sent_num == len(graph), "graph not correct!"
        assert count_sent_num == count_graph_num, "label not correct!"
        datasets.append(b_data_dict)

    print(len(datasets))
    print('Saving to %s' % save_file)
    # print('total_docs:%s    total_sens:%s    toatl_words:%s' % (total_docs, total_sens, total_words))
    # print('#sen/doc:%s    #word/doc:%s    #word/sen:%s' % (
    # total_sens / total_docs, total_words / total_docs, total_words / total_sens))
    # print('The ratio of similarity larger than %s is %s' % (args.sim_threshold, total_larger / (total_sum + 1e-18)))
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

    total_sent_list = []
    for doc in corpus:
        for para in doc:
            total_sent_list.append(para)
    total_sent_num = len(total_sent_list)

    sim_graph = []
    raw_corpus = [para for para in total_sent_list]

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

    # print("sim_graph[0]: %s" % str(sim_graph[0]))

    # trans sim_graph into doc format, more convient for padding process
    """
    before trans: total_sent_num * total_sent_num
    after trans: doc_num * each_doc_sent_num * total_sent_num
    """
    print("total sent num: " + str(total_sent_num))
    doc_sim_graph = []
    cur_sent_id = 0
    for doc in corpus:
        cur_doc_sent_num = len(doc)
        cur_doc_sim_graph = sim_graph[cur_sent_id:cur_doc_sent_num+cur_sent_id]
        # cur_doc_sim_graph = [graph[cur_sent_id:total_sent_num+cur_sent_id] for graph in cur_doc_sim_graph]
        doc_sim_graph.append(cur_doc_sim_graph)
        cur_sent_id += cur_doc_sent_num

    return doc_sim_graph, count_large, total


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
        sim_graph.append(sim.tolist())

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
        sim_graph.append(sim.tolist())

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

        self.tokenizer = GptBpeTokenizer(vocab_file=args.roberta_vocab_file,
                                         encoder_json_file=args.encoder_json_file,
                                         vocab_bpe_file=args.vocab_bpe_file,
                                         do_lower_case=True)

        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.unk_token = self.tokenizer.unk_token
        self.pad_token = self.tokenizer.pad_token
        self.mask_token = self.tokenizer.mask_token

        self.tgt_bos = self.cls_token
        self.tgt_eos = self.mask_token
        self.tgt_sent_split = self.sep_token

        self.sep_vid = self.tokenizer.sep_token_id
        self.cls_vid = self.tokenizer.cls_token_id
        self.pad_vid = self.tokenizer.pad_token_id

        # vocab convert wikisum idx to tokens
        """load vocabulary"""
        self.spm = sentencepiece.SentencePieceProcessor()
        self.spm.load(args.vocab_path)
        self.vocab_size = len(self.spm)

    def preprocess(self, srcs, tgt, args):
        """preprocess src and tgt by tokenization"""

        docs_src_token_ids = []
        docs_src = []
        docs_sent_labels = []
        docs_cls_ids = []
        docs_sep_ids = []
        docs_sent_rank = []
        docs_summary_rank = []
        docs_positive_sent_idx = []
        docs_negative_sent_idx = []
        docs_source = []

        for src in srcs:
            if len(src) == 0:
                # return None
                continue
            idxs = [i for i, s in enumerate(src) if (len(s.split(" ")) > self.args.min_src_ntokens)]
            # idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]
            # print("sent_labels: " + str(sent_labels) + " len: " + str(len(sent_labels)))
            source = src[:]
            target = tgt[:]
            src = [src[i] for i in idxs]
            src = src[:self.args.max_nsents]
            docs_source += src


            assert len(src) <= self.args.max_nsents, "docs filtered wrong!!!"

            # discard too short documents
            if len(src) < self.args.min_nsents:
                # return None
                continue

            src_token_ids = []
            for sent in src:
                cur_sent_token = [self.cls_token] + self.tokenizer.tokenize(sent) + [self.sep_token]
                cur_sent_token_ids = self.tokenizer.convert_tokens_to_ids(cur_sent_token)
                src_token_ids.extend(cur_sent_token_ids)

            cls_ids = [i for i, t in enumerate(src_token_ids) if t == self.cls_vid]
            sep_ids = [i for i, t in enumerate(src_token_ids) if t == self.sep_vid]

            true_sent_num = len([id for id in cls_ids if id < self.args.max_src_ntokens])
            # print("true_sent_num: " + str(true_sent_num))

            if args.label_selection_type == "greedy_selection":
                sent_labels = _greedy_selection(source, target, args, true_sent_num)
                sent_rouges = _rouge_selection(source, target, args, true_sent_num)
            elif args.label_selection_type == "rouge_selection":
                sent_labels = _rouge_selection(source, target, args, true_sent_num)
                sent_rouges = sent_labels
            else:
                raise ValueError("The label selection has been set wrong!")

            labeled_sent_ids = sent_labels[:]
            origin_labels = sent_labels[:]
            # print("origin labels: " + str(origin_labels))
            _sent_labels = [0] * len(source)
            # print("origin sent num: ", len(source))
            if args.label_selection_type == "greedy_selection":
                for l in sent_labels:
                    _sent_labels[l] = 1
                sent_labels = [_sent_labels[i] for i in idxs]
                sent_labels = sent_labels[:self.args.max_nsents]

                sent_rouges = [sent_rouges[i] for i in idxs]
                sent_rouges = sent_rouges[:self.args.max_nsents]
                origin_sent_rank = np.argsort(-np.array(sent_rouges, dtype = float)).tolist()
                sent_rank = [rank for rank in origin_sent_rank if sent_rouges[rank] != 0]
                summary_rank = _get_summary_rank(source, target, sent_labels, origin_sent_rank, self.args, true_sent_num)
                # print("summary_rank num: " + str(len(summary_rank)))
                for summary in summary_rank:
                    for sent_id in summary:
                        if sent_id not in idxs:
                            summary.remove(sent_id)
                positive_sent_idx = [idx for idx, label in enumerate(sent_labels) if label == 1]
                negative_sent_idx = [idx for idx, label in enumerate(sent_labels) if label == 0]
            elif args.label_selection_type == "rouge_selection":
                sent_labels = [sent_labels[i] for i in idxs]
                sent_labels = sent_labels[:self.args.max_nsents]
                # print("sent labels: " + str(sent_labels))
                sent_rank = np.argsort(-np.array(sent_labels, dtype = float)).tolist()
                sent_rank = [rank for rank in sent_rank if sent_labels[rank] != 0]
                # print("sent rank: " + str(sent_rank))
            else:
                raise ValueError("The label selection has been set wrong!")
            
            assert len(src) == len(sent_labels), "sent labels not equal with sent num!!!"
            assert len(sent_labels) == len(cls_ids), "sent labels not equal with cls ids!!!"

            docs_src_token_ids.append(src_token_ids)
            docs_src.append(src)
            docs_sent_labels.append(sent_labels)
            docs_cls_ids.append(cls_ids)
            docs_sep_ids.append(sep_ids)
            docs_sent_rank.append(origin_labels)
            docs_summary_rank.append(summary_rank)
            docs_positive_sent_idx.append(positive_sent_idx)
            docs_negative_sent_idx.append(negative_sent_idx)

        tgt_sents = nltk.tokenize.sent_tokenize(tgt.strip())
        tgt_txt = tgt_sents
        tgt_token = [self.cls_token] + self.tokenizer.tokenize(tgt) + [self.sep_token]
        tgt_token_ids = self.tokenizer.convert_tokens_to_ids(tgt_token)

        true_sent_labels = _multidoc_greedy_selection(docs_source, tgt, args)
        if len(true_sent_labels) == 0:
            return None
        if len([i for i in true_sent_labels if i == 1]) == 0:
            return None
        multidoc_sent_labels = []
        for doc in docs_cls_ids:
            cur_doc_sent_num = len(doc)
            cur_doc_sent_label = true_sent_labels[:cur_doc_sent_num]
            del true_sent_labels[:cur_doc_sent_num]
            multidoc_sent_labels.append(cur_doc_sent_label)

        res = namedtuple('result', ['src_token_ids', 'tgt_token_ids', 'src_txt', 'tgt_txt', 'sent_labels', 'cls_ids', 'sep_ids', 
                                'sent_rank', 'summary_rank', 'positive_sent_idx', 'negative_sent_idx'])
        # print("sent num: %d,  label num: %d" % (len(docs_src_token_ids), len(docs_sent_labels)))

        return res(src_token_ids=docs_src_token_ids, tgt_token_ids=tgt_token_ids, src_txt=docs_src, tgt_txt=tgt_txt, 
                    sent_labels=multidoc_sent_labels, cls_ids=docs_cls_ids, sep_ids=docs_sep_ids, 
                    sent_rank=docs_sent_rank, summary_rank=docs_summary_rank,
                    positive_sent_idx=docs_positive_sent_idx, negative_sent_idx=docs_negative_sent_idx)
        
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


def _greedy_selection(doc_sent_list, tgt_str, args, true_sent_num):

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def _cal_rouge(evaluated_ngrams, reference_ngrams):
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return {"f": f1_score, "p": precision, "r": recall}

    sent_str_list = doc_sent_list[:args.max_nsents]
    doc_txt = " ".join(sent_str_list)
    doc_word_list = doc_txt.split(" ")[:args.max_src_ntokens]
    doc_txt = " ".join(doc_word_list)
    sent_str_list = nltk.tokenize.sent_tokenize(doc_txt.strip())
    sent_str_list = doc_sent_list
    max_rouge = 0.0
    abstract = _rouge_clean(tgt_str).split()
    sents = [_rouge_clean(s).split() for s in sent_str_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for _ in range(args.summary_sent_num):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = _cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = _cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            selected = [id for id in selected if id < true_sent_num]
            return sorted(selected)
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    selected = [id for id in selected if id < true_sent_num]
    return sorted(selected)

def _rouge_selection(doc_sent_list, tgt_str, args, true_sent_num):

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def _cal_rouge(evaluated_ngrams, reference_ngrams):
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return {"f": f1_score, "p": precision, "r": recall}

    sent_str_list = doc_sent_list[:true_sent_num]
 
    max_rouge = 0.0
    abstract = _rouge_clean(tgt_str).split()
    sents = [_rouge_clean(s).split() for s in sent_str_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    rouge_scores = []
    for i in range(len(sents)):
        candidates_1 = [evaluated_1grams[i]]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[i]]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = _cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = _cal_rouge(candidates_2, reference_2grams)['f']
        rouge_score = (rouge_1 + rouge_2) / 2
        rouge_scores.append(rouge_score)

    rouge_scores = rouge_scores + [0.] * (len(doc_sent_list) - len(sent_str_list))
    return rouge_scores

def _get_summary_rank(doc_sent_list, tgt_str, labeled_sent_ids, sent_rank, args, true_sent_num):
    
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def _cal_rouge(evaluated_ngrams, reference_ngrams):
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return {"f": f1_score, "p": precision, "r": recall}
    
    def _cal_summary_rouges(candidate_summary_list, evaluated_1grams, evaluated_2grams, reference_1grams, reference_2grams):
        summary_rouge_scores = []
        for c in candidate_summary_list:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = _cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = _cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = (rouge_1 + rouge_2) / 2
            summary_rouge_scores.append(rouge_score)
        return summary_rouge_scores

    summary_rank = []
    sent_str_list = doc_sent_list[:true_sent_num]

    abstract = _rouge_clean(tgt_str).split()
    sents = [_rouge_clean(s).split() for s in sent_str_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])
    tmp_sent_ids = []
    for sent_id in sent_rank:
        if sent_id not in labeled_sent_ids and sent_id < true_sent_num and len(sent_str_list[sent_id]) > args.min_src_ntokens:
            tmp_sent_ids.append(sent_id)
    labeled_sent_ids = tmp_sent_ids[:5]
    candidate_summary_list = list(itertools.combinations(labeled_sent_ids, 3))
    if len(candidate_summary_list) != 10:
        return []
    candidate_summary_rouge_scores = _cal_summary_rouges(candidate_summary_list, evaluated_1grams, 
            evaluated_2grams, reference_1grams, reference_2grams)
    candidate_summary_rank = np.argsort(-np.array(candidate_summary_rouge_scores, dtype = float)).tolist()
    for c_id in candidate_summary_rank:
        summary_rank.append(list(candidate_summary_list[c_id]))
    return summary_rank
    
def _multidoc_greedy_selection(doc_sent_list, tgt_str, args):

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def _cal_rouge(evaluated_ngrams, reference_ngrams):
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return {"f": f1_score, "p": precision, "r": recall}

    sent_str_list = doc_sent_list
    sent_labels = [0] * len(doc_sent_list)

    max_rouge = 0.0
    abstract = _rouge_clean(tgt_str).split()
    sents = [_rouge_clean(s).split() for s in sent_str_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for _ in range(args.summary_sent_num):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = _cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = _cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            # selected = [id for id in selected if id < true_sent_num]
            for id in selected:
                sent_labels[id] = 1
            # print("final rouge: " + str(max_rouge))
            return sent_labels
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    # selected = [id for id in selected if id < true_sent_num]
    for id in selected:
        sent_labels[id] = 1
    return sent_labels

def cal_multidoc_sent_num(multidoc_data):
    sent_num = 0
    for doc in multidoc_data:
        sent_num += len(doc)
    return sent_num