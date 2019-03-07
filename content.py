# -*- coding: UTF-8 -*-
'''
文本的预处理：
1、繁体字转化为简体字。
2、去掉空格，标点符号以及表情符号，得到纯文本。
3、利用jieba进行分词，分词之后去掉停用词。
4、保存结果至本地，'content.csv'
'''
import pandas as pd
import re
from langconv import *
import jieba

jieba.initialize()


def textHandler(sentence):
    sentence = chs_to_cht(sentence)
    sentence = remove_pun(sentence)
    words = jieba.lcut(sentence)
    result = ''
    for word in words:
        result = result + word + ' '
    result = result + '\n'
    return result


def chs_to_cht(sentence):
    sentence = Converter('zh-hans').convert(sentence)
    sentence.encode('utf-8')
    return sentence


def remove_pun(sentence):
    # 可以去掉空格，标点，颜文字，\n,\t,\r
    return re.sub(u'[^\u4e00-\u9fa5]', '', sentence)


data = pd.read_csv('test.csv')
content = data['content']
file = open('test_content.txt', 'w', encoding='utf-8-sig')
for index in range(len(content)):
    text = content[index]
    text = textHandler(text)
    file.write(text)
    if index % 1000 == 0:
        print(index)
file.close()
