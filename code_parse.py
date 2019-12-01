#!/usr/bin/env python
# encoding: utf-8
'''
@author: leexuan
@contact: xuanli19@fudan.edu.cn
@Time : 2019-10-28 19:52
@desc: 
'''

import plyj.parser  # pip install plyj
import glob
import csv
from ply import lex, yacc
import pickle


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


if __name__ == '__main__':
    # 完成代码统计任务 ，并存到文件中
    parser = Parser()
    lis = glob.glob('../ASE_javacode/*')
    di = dict()
    for i in range(len(lis)):
        print(f'{i}/{len(lis)}  {lis[i]}')
        try :
            res = parser.tokenize_file(lis[i])
        except Exception :
            continue
        for i in range(len(res)):
            # print(res[i],res[i].type, res[i].value, res[i].lineno, res[i].lexpos ) # NUM  和 STRING_LITERAL
            if res[i].type == 'NUM':
                # print(res[i], res[i].type, res[i].value, res[i].lineno, res[i].lexpos)  # NUM  和 STRING_LITERAL

                di['<num>'] = di.get('<num>',0)+1
            elif res[i].type == 'STRING_LITERAL':
                di['<str>'] = di.get('<str>',0)+1
            else:
                di[res[i].value]=di.get(res[i].value,0)+1
        print('dict len', len(di))

    # with open('all_word.pkl', 'wb')as f:
    #     pickle.dump(di, f)

