# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import unicodedata
import six
import pickle
import nltk
import numpy as np
import random
import re

from utils import common_lib


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "rb") as fin:
        for num, line in enumerate(fin):
            items = convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def basic_tokenizer(sen):
    """doc"""
    seg = sen.split(b' ')
    seg = filter(lambda i: i != b' ', seg)
    return seg


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def __call__(self, sen):
        return self.tokenize(sen)

    def tokenize(self, text):
        """
            tokenize
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """
            convert_tokens_to_ids
        """
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        """
            convert_ids_to_tokens
        """
        return convert_by_vocab(self.inv_vocab, ids)


class CharTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def __call__(self, sen):
        return self.tokenize(sen)

    def tokenize(self, text):
        """
            tokenize
        """
        split_tokens = []
        for token in text.lower().split(" "):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """
            convert_tokens_to_ids
        """
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        """
            convert_ids_to_tokens
        """
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def __call__(self, sen):
        return self.tokenize(sen)

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def __call__(self, sen):
        return self.tokenize(sen)

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.

        Returns:
            A list of wordpiece tokens.
        """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def merge_subword(self, tokens):
        """
        :param tokens:
        :return: merged_tokens
        """
        ret = []
        for token in tokens:
            if token.startswith("##"):
                real_token = token[2:]
                if len(ret):
                    ret[-1] += real_token
                else:
                    ret.append(real_token)
            else:
                ret.append(token)
        return ret


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or _is_whitespace(char):
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def wordpiece(token, vocab, unk_token, sentencepiece_style_vocab=False, max_input_chars_per_word=100):
    """call with single word"""
    chars = list(token)
    if len(chars) > max_input_chars_per_word:
        return [unk_token], [(0, len(chars))]

    is_bad = False
    start = 0
    sub_tokens = []
    sub_pos = []
    while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
            substr = "".join(chars[start:end])
            if start == 0 and sentencepiece_style_vocab:
                substr = u'\u2581' + substr
            if start > 0 and not sentencepiece_style_vocab:
                substr = "##" + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            is_bad = True
            break
        sub_tokens.append(cur_substr)
        sub_pos.append((start, end))
        start = end
    if is_bad:
        return [unk_token], [(0, len(chars))]
    else:
        return sub_tokens, sub_pos


class SpaceTokenizer(object):
    """
        char tokenizer (wordpiece english)
        normed txt(space seperated or not) => list of word-piece
    """
    def __init__(self, vocab, lower=True):
        self.vocab = set(vocab)
        self.lower = lower

    def __call__(self, sen):
        if len(sen) == 0:
            return [] #empty line
        sen = sen.decode('utf8')
        if self.lower:
            sen = sen.lower()
        res = []
        for s in sen.split(' '):
            if s == ' ':
                continue
            if s in self.vocab:
                res.append(s)
            else:
                res.append('[UNK]')
        return res


class CharSpTokenizer(object):
    """
        char tokenizer (wordpiece english)
        normed txt(space seperated or not) => list of word-piece
    """
    def __init__(self, vocab, lower=True, sentencepiece_style_vocab=False): 
        self.vocab = set(vocab)
        #self.pat = re.compile(r'([,.!?\u3002\uff1b\uff0c\uff1a\u201c\u201d\
        #   \uff08\uff09\u3001\uff1f\u300a\u300b]|[\u4e00-\u9fa5]|[a-zA-Z0-9]+)')
        self.pat =  re.compile(r'([a-zA-Z0-9]+|\S)')
        self.lower = lower
        self.sentencepiece_style_vocab = sentencepiece_style_vocab

    def __call__(self, sen):
        if len(sen) == 0:
            return [] #empty line
        sen = sen.decode('utf8')
        if self.lower:
            sen = sen.lower()
        res = []
        for match in self.pat.finditer(sen):
            words, _ = wordpiece(match.group(0), vocab=self.vocab, unk_token='[UNK]',
                    sentencepiece_style_vocab=self.sentencepiece_style_vocab)
            res.extend(words)
        return res


class WSSPTokenizer(object):
    """
        WSSPTokenizer
    """
    def __init__(self, sp_model_file, word_dict, ws=True, lower=True):
        self.ws = ws
        self.lower = lower
        self.dict = pickle.load(open(word_dict, 'rb', encoding='utf8'))
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.window_size = 5
        self.sp_model.Load(sp_model_file)

    def _cut(self, chars):
        words = []
        idx = 0
        while idx < len(chars):
            matched = False
            for i in range(self.window_size, 0, -1):
                cand = chars[idx: idx + i]
                if cand in self.dict:
                    words.append(cand)
                    matched = True
                    break
            if not matched: 
                i = 1
                words.append(chars[idx])
            idx += i
        return words
 
    def __call__(self, sen):
        sen = sen.decode('utf8')
        if self.ws:
            sen = [s for s in self._cut(sen) if s != ' ']
        else:
            sen = sen.split(' ')
        if self.lower:
            sen = [s.lower() for s in sen]
        sen = ' '.join(sen)
        ret = self.sp_model.EncodeAsPieces(sen)
        return ret
    
    def merge_subword(self, tokens):
        """
        :param tokens:
        :return: merged_tokens
        """
        ret = []
        for token in tokens:
            if token.startswith(u"▁"):
                ret.append(token[1:])
            else:
                if len(ret):
                    ret[-1] += token
                else:
                    ret.append(token)

        ret = [token for token in ret if token]
        return ret


def build_2_pair(seg_a, seg_b, max_seqlen, cls_id, sep_id):
    """
        build pair for two sentence
    """
    truncate_seq_pair(seg_a, seg_b, max_seqlen - 3) 
    
    sen_emb = np.concatenate([[cls_id], seg_a, [sep_id], seg_b, [sep_id]], 0)
    token_type_a = np.ones_like(seg_a, dtype=np.int64) * 0
    token_type_b = np.ones_like(seg_b, dtype=np.int64) * 1
    token_type_emb = np.concatenate([[0], token_type_a, [0], token_type_b, [1]], 0)

    return sen_emb, token_type_emb


def build_1_pair(seg_a, max_seqlen, cls_id, sep_id):
    """
        build pair for one sentence
    """
    seg_a = truncate_words(seg_a, max_seqlen - 2, "KEEP_HEAD")

    token_type_a = np.ones_like(seg_a, dtype=np.int64) * 0
    sen_emb = np.concatenate([[cls_id], seg_a, [sep_id]], 0)
    token_type_emb = np.concatenate([[0], token_type_a, [0]], 0)
    
    return sen_emb, token_type_emb


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
        Truncates a sequence pair in place to the maximum length.
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def truncate_multi_seqs(tokens_arr, max_length):
    """
        truncate_seqs
    """
    while True:
        ls = [len(ts) for ts in tokens_arr]
        total_length = sum(ls)
        if total_length <= max_length:
            break
        max_l = max(ls)
        ind = ls.index(max_l)
        trunc_tokens = tokens_arr[ind]

        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

 
def truncate_words(words, max_length, truncation_type):
    """
    :param words:
    :param max_length:
    :param truncation_type:
    :return:
    """
    if len(words) > max_length:
        if truncation_type == "KEEP_HEAD":
            words = words[0: max_length]
        elif truncation_type == "KEEP_TAIL":
            tmp = words[0: max_length - 1]
            tmp.append(words[-1])
            words = tmp
        elif truncation_type == "KEEP_BOTH_HEAD_TAIL":
            tmp = words[1: max_length - 2]
            tmp.insert(0, words[0])
            tmp.insert(max_length - 1, words[-1])
            words = tmp
        else:
            words = words[0: max_length]

    return words


def expand_dims(*args):
    """
        expand_dims
    """
    func = lambda i: np.expand_dims(i, -1)
    ret = [func(i) for i in args]
    return ret


def mixed_segmentation(in_str, rm_punc=False):
    """
        # split Chinese with English
    """
    in_str = in_str.lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = [
        '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：',
        '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「', '」', '（',
        '）', '－', '～', '『', '』'
    ]
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def remove_punctuation(in_str):
    """
        remove punctuation
    """
    in_str = in_str.lower().strip()
    sp_char = [
        '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：',
        '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「', '」', '（',
        '）', '－', '～', '『', '』'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def calc_f1_score(answers, prediction):
    """
        mrc calc_f1_score
    """
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = common_lib.find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    """
        mrc calc_em_score
    """
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def gen_unidirectional_mask(insts, sent_b_starts=None):
    """
    generate input mask for seq2seq
    """
    max_len = max(len(inst) for inst in insts)
    input_mask_data = np.zeros((len(insts), max_len, max_len))
    for index, mask_data in enumerate(input_mask_data):
        start = sent_b_starts[index]
        end = len(insts[index])
        mask_data[:end, :start] = 1.0
        # Generate the lower triangular matrix using the slice of matrix
        b = np.tril(np.ones([end - start, end - start]), 0)
        mask_data[start:end, start:end] = b
    input_mask_data = np.array(input_mask_data, dtype='float32').reshape([-1, max_len, max_len])
    return input_mask_data


def gen_query_input(token_ids, max_len, sent_b_starts, mask_id):
    """
    generate query input when using two-stream
    """
    bsz = len(sent_b_starts)
    dec_len = [len(token_ids[i]) - sent_b_starts[i] for i in range(bsz)]
    max_len_query = max(dec_len)
    mask_datas = np.zeros((bsz, max_len_query, max_len + max_len_query))
    mask_ids = np.ones((bsz, max_len_query, 1)) * mask_id
    tgt_pos = []
    for i in range(bsz):
        tgt_pos.extend(list(range(max_len_query * i + 1, max_len_query * i + dec_len[i])))
    for index, mask_data in enumerate(mask_datas):
        for i in range(dec_len[index]):
            mask_data[i, :sent_b_starts[index] + i] = 1.0
            mask_data[i, max_len + i] = 1.0

    return (mask_datas.astype('float32'),
           mask_ids.astype('int64'),
           np.array(tgt_pos).reshape([-1, 1]).astype('int64'))


def is_alphabet_or_digit(c):
    """
    abc or 123
    """
    alphabet = list(u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digit = list(u"0123456789.")
    if c in alphabet or c in digit:
        return True
    return False
 

