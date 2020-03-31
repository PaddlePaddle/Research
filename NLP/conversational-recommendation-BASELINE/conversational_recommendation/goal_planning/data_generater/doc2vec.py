#!/usr/bin/python
# -*- coding: UTF-8 -*-
from gensim import corpora, models, similarities
from collections import defaultdict
import jieba
import re
jieba.load_userdict("../origin_data/user_dict.txt")

class Doc2Vec(object):
    def __init__(self, documents):
        self.stop_words = self.stop_words()
        self.word_dict, self.texts = self.dictionary_generator(documents)
        self.UNK = self.word_dict["unk"]

    def stop_words(self):
        stop_words = list()
        with open("../origin_data/stop_words.txt", "r") as f:
            for line in f.readlines():
                stop_words.append(line.replace("\n", ""))
        return stop_words

    def text_generator(self, documents):
        texts_idx = list()
        for doc in documents:
            doc_idx = list()
            for line in doc:
                line_idx = list()
                line = self.remove_punctuation(line)
                words = ' '.join(jieba.cut(line)).split(' ')
                for word in words:
                    line_idx.append(self.word_dict.get(word, self.UNK))
                doc_idx.append(line_idx)
            texts_idx.append(doc_idx)
        return texts_idx

    def dictionary_generator(self, documents):
        documents = [self.remove_punctuation(line) for doc in documents for line in doc]
        texts = list()
        for line in documents:
            text= list()
            words = ' '.join(jieba.cut(line)).split(' ')
            for word in words:
                # if word not in self.stop_words:
                text.append(word)
            texts.append(text)
        
        frequency = defaultdict(int)
        for text in texts:
            for word in text:
                frequency[word] += 1
        docs = [[word for word in text if frequency[word] > 5] for text in texts]
        dictionary = corpora.Dictionary(docs)
        corpus = [dictionary.doc2bow(text) for text in docs]
        word_dict = dict()
        # word_dict["pad"] = 0
        for k, v in dictionary.items():
            word_dict[v] = k
        word_dict["unk"] = len(word_dict)
        # for k, v in word_dict.items():
        #     print(k, v)
        print(len(word_dict))
        return word_dict, texts

    def remove_punctuation(self, line):
        line = re.sub("\[\d*\]", "", line)
        return re.sub('[^\u4e00-\u9fa5^a-z^A-Z^0-9]', '', line)

    
