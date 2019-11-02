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
    def __init__(self, data_path='../ASE_javacode/*'):
        parser = Parser()
        lis = glob.glob(data_path)
        print(lis[0])
        res = parser.tokenize_file(lis[0])
        for i in range(len(res)):
            # print(res[i],res[i].type, res[i].value, res[i].lineno, res[i].lexpos ) # NUM  和 STRING_LITERAL
            if res[i].type == 'NUM' or res[i].type == 'STRING_LITERAL':
                print(res[i], res[i].type, res[i].value, res[i].lineno, res[i].lexpos)  # NUM  和 STRING_LITERAL


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
