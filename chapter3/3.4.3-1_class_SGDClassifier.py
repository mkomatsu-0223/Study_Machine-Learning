# -*- coding: utf-8 -*-
from sklearn.linear_model import SGDClassifier

# 確率的勾配降下法バージョンのパーセプトロンを生成
ppn = SGDClassifier(loss='perceptron')
# 確率的勾配降下法バージョンのロジスティック回帰を生成
lr = SGDClassifier(loss='log')
# 確率的勾配降下法バージョンのSVM（損失関数＝ヒンジ）を生成
svm = SGDClassifier(loss='hinge')
