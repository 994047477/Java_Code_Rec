#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019-11-02 18:49
@desc: 
'''
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import plyj.parser  # pip install plyj
import glob
from ply import lex, yacc
import pickle
from collections import Counter

import config


def read_data_counter(file_path='all_word_1000.pkl'):
    with open(file_path, 'rb') as f:
        di = pickle.load(f)
    return di


class Parser(object):

    def __init__(self):
        self.lexer = lex.lex(module=plyj.parser.MyLexer(), optimize=1)
        self.parser = yacc.yacc(module=plyj.parser.MyParser(), start='goal', optimize=1)

    def tokenize_string(self, code):
        self.lexer.input(code)
        lis = []
        for token in self.lexer:
            lis.append(token)
        return lis

    def tokenize_file(self, _file):
        if type(_file) == str:
            _file = open(_file)
        content = ''
        for line in _file:
            content += line
        return self.tokenize_string(content)

    def parse_expression(self, code, debug=0, lineno=1):
        return self.parse_string(code, debug, lineno, prefix='--')

    def parse_statement(self, code, debug=0, lineno=1):
        return self.parse_string(code, debug, lineno, prefix='* ')

    def parse_string(self, code, debug=0, lineno=1, prefix='++'):
        self.lexer.lineno = lineno
        return self.parser.parse(prefix + code, lexer=self.lexer, debug=debug)

    def parse_file(self, _file, debug=0):
        if type(_file) == str:
            _file = open(_file)
        content = ''
        for line in _file:
            content += line
        return self.parse_string(content, debug=debug)


class CodeDataset(Dataset):
    def __init__(self, data_path='../ASE_javacode/*',data_num=100):
        parser = Parser()
        lis = glob.glob(data_path)
        self.All_sentences = []
        self.All_next_word = []
        di = read_data_counter()
        di_5 = set([k for k, v in di.items() if v > 5])
        di_5.add('<unk>')

        words = sorted(list(di_5))
        self.word_indices = dict((word, index) for index, word in enumerate(words))
        # print(self.word_indices)
        # torch.nn.functional.one_hot()
        # 不能使用eye 生成  会爆内存
        
        maxlen = config.max_size

        for j in range(len(lis)):
            print(f'{j}/{len(lis)}  {lis[j]}')
            try:
                all_words = parser.tokenize_file(lis[j])
                all_words1 = []
                for word in all_words:
                    if word.type == 'NUM':
                        all_words1.append('<num>')
                    elif word.type == 'STRING_LITERAL':
                        all_words1.append('<str>')
                    elif word.value not in di_5:
                        all_words1.append('<unk>')
                    else:

                        all_words1.append(word.value)
            except Exception:
                continue
            for i in range(0, len(all_words1) - maxlen):
                self.All_sentences.append(all_words1[i: i + maxlen])
                self.All_next_word.append(all_words1[i+1:i + maxlen+1])
                # print('test',all_words1[i: i + maxlen],all_words1[i + maxlen])

            if j == data_num:
                break

    def __getitem__(self, index):
        # 返回的时候才将词转换成 embedding 放序号就行 ，不用转换成one-hot encoding
        # return self.All_sentences[index], \
        #        self.All_next_word[index]

        return torch.tensor([ self.word_indices[i] for i in self.All_sentences[index]]),\
               torch.tensor([ self.word_indices[i] for i in self.All_next_word[index]])

    def __len__(self):
        return len(self.All_next_word)
if __name__ == '__main__':
    dataset = CodeDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=32)
    # for i, (X, Y) in enumerate(dataloader):
    #     print(i, (X, Y))
    print( np.shape(dataset.All_sentences) ,np.shape(dataset.All_next_word))
    print(dataset.All_sentences[0])




