# coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import json
import os

from paddlehub.common.logger import logger
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset
from paddlehub.dataset.cmrc2018 import CMRC2018Example
from paddlehub.reader import tokenization

SPIECE_UNDERLINE = '▁'


class DuReader(BaseNLPDataset):
    """
    2020语言与智能技术竞赛：机器阅读理解任务数据集
    """
    def __init__(self):
        dataset_dir = './dureader_robust-data'
        super(DuReader, self).__init__(
            base_path=dataset_dir,
            train_file="train.json",
            dev_file="dev.json",
            predict_file='test1.json'
        )

    def _read_file(self, input_file, phase=False):
        """
        读入json格式数据集
        """
        def _is_chinese_char(cp):
            if ((cp >= 0x4E00 and cp <= 0x9FFF)
                    or (cp >= 0x3400 and cp <= 0x4DBF)
                    or (cp >= 0x20000 and cp <= 0x2A6DF)
                    or (cp >= 0x2A700 and cp <= 0x2B73F)
                    or (cp >= 0x2B740 and cp <= 0x2B81F)
                    or (cp >= 0x2B820 and cp <= 0x2CEAF)
                    or (cp >= 0xF900 and cp <= 0xFAFF)
                    or (cp >= 0x2F800 and cp <= 0x2FA1F)):
                return True
            return False

        def _is_punctuation(c):
            if c in [
                    '。', '，', '！', '？', '；', '、', '：', '（', '）', '－', '~', '「',
                    '《', '》', ',', '」', '"', '“', '”', '$', '『', '』', '—', ';',
                    '。', '(', ')', '-', '～', '。', '‘', '’', '─', ':'
            ]:
                return True
            return False

        def _tokenize_chinese_chars(text):
            """
            中文汉字切分
            """
            output = []
            for char in text:
                cp = ord(char)
                if _is_chinese_char(cp) or _is_punctuation(char):
                    if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                        output.append(SPIECE_UNDERLINE)
                    output.append(char)
                    output.append(SPIECE_UNDERLINE)
                else:
                    output.append(char)
            return "".join(output)

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(
                    c) == 0x202F or ord(c) == 0x3000 or c == SPIECE_UNDERLINE:
                return True
            return False

        examples = []
        drop = 0
        with open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                context = _tokenize_chinese_chars(paragraph_text)

                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in context:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    if c != SPIECE_UNDERLINE:
                        char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    
                    
                    if phase == 'predict':
                        # 测试集部分没有答案
                        orig_answer_text = ""
                        start_position=-1
                        end_position=-1
                    else:
                        # 训练集/验证集选择第一个答案作为ground truth
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        while paragraph_text[answer_offset] in [
                                " ", "\t", "\r", "\n", "。", "，", "：", ":", ".", ","
                        ]:
                            answer_offset += 1
                        start_position = char_to_word_offset[answer_offset]
                        answer_length = len(orig_answer_text)

                        end_offset = answer_offset + answer_length - 1
                        if end_offset >= len(char_to_word_offset):
                            end_offset = len(char_to_word_offset) - 1
                        end_position = char_to_word_offset[end_offset]

                    if phase == "train":
                        actual_text = "".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = "".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            drop += 1
                            logger.warning((actual_text, " vs ",
                                            cleaned_answer_text, " in ", qa))
                            continue
                    example = CMRC2018Example(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position)
                    examples.append(example)

        logger.warning("%i bad examples has been dropped" % drop)
        return examples


if __name__ == "__main__":
    ds = DuReader()
    print("train")
    examples = ds.get_train_examples()
    for index, e in enumerate(examples):
        if index < 2:
            print(e)
    print("dev")
    examples = ds.get_dev_examples()
    for index, e in enumerate(examples):
        if index < 2:
            print(e)
            
    print("predict")
    examples = ds.get_predict_examples()
    for index, e in enumerate(examples):
        if index < 2:
            print(e)
