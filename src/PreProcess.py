# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
import os
import jieba
import jieba.analyse

from sklearn.model_selection import train_test_split

# 一共有6种类别：'Military', 'Economy', 'Culture', 'Sports', 'Auto', 'Medicine'

class Data(object):
    def __init__(self, data_path, stop_file=None):
        if not os.path.exists("../data/data.pkl"):
            self.data, self.label = [], []
            allfile = os.listdir(path=data_path)
            for file in allfile:
                label = file.split('_')[0]
                filePath = os.path.join(data_path, file)
                with open(filePath, 'r', encoding='utf-8') as fd:
                    context = fd.read()
                self.data.append(context)
                self.label.append(label)
            self.data, self.label = self.delete_and_split(self.data, self.label)
            pickle.dump((self.data, self.label), open("../data/data.pkl", "wb"))
        else:
            self.data, self.label = pickle.load(open("../data/data.pkl", "rb"))

        if stop_file is not None:
            self.stopword = set()
            fd = open(stop_file, 'r', encoding='utf-8')
            for line in fd:
                self.stopword.add(line.strip())

    @staticmethod
    def delete_and_split(data, label):
        new_data = []
        data = zip(data, label)
        for context, label in data:
            string = ' '.join(list(jieba.cut(context)))
            new_data.append((string, label))
        return zip(*new_data)

    @staticmethod
    def remove_stop_word(article, stopword):
        '''
        Data当中没有使用，可以考虑怎么使用
        :param article: 文档
        :return: list，返回删除停用词的文档
        '''
        new_article = []
        for word in article:
            if word not in stopword:
                new_article.append(word)
        return new_article
