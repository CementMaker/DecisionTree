# -*- coding:utf-8 -*-
# !/usr/bin/env python

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from PreProcess import *

class decisionTreeClassify(object):
    def __init__(self):
        data = Data("../data/corpus")
        self.data, self.label = data.data, data.label
        # 分割数据集
        train_data, test_data, train_label, test_label = train_test_split(self.data, self.label, test_size=0.2)

        # 此处的CountVectorizer也可以使用TfidfVectorizer
        # 参考文档CountVectorizer：http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        # 参考文档TfidfVectorizer：http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        vectorizer = CountVectorizer(max_df=0.5,
                                     max_features=5000,
                                     min_df=2,
                                     lowercase=False,
                                     decode_error='ignore',
                                     analyzer=str.split).fit(train_data)
        # 转化成词袋模型向量
        self.train_x = vectorizer.transform(train_data)
        self.test_x = vectorizer.transform(test_data)

        # 定义一个决策树
        # 参考文档：http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        DecisionTree = DecisionTreeClassifier(criterion="entropy",
                                              splitter="best",
                                              max_depth=None,
                                              min_samples_split=2,
                                              min_samples_leaf=2)
        # 训练加上测评
        DecisionTree.fit(self.train_x, train_label)
        print("分类的准确率为：", DecisionTree.score(self.test_x, test_label))

        # 特征提取，提取词汇
        words = vectorizer.get_feature_names() #获取词袋模型的词汇
        feature_importance = DecisionTree.feature_importances_ # 决策树进行特征提取的特征的重要性
        word_importances_dict = dict(zip(words, feature_importance))

        # 输出前200个重要的词汇
        number = 0
        for word, importance in sorted(word_importances_dict.items(), key=lambda val: val[1], reverse=True):
            print(word, importance)
            number += 1
            if number == 200:
                break

if __name__ == "__main__":
    decisionTree = decisionTreeClassify()