# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:54:13 2021

@author: KOMATSU
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Irisデータをロード
iris = datasets.load_iris()

# 3、4列目の特徴量を抽出
X = iris.data[:, [2, 3]]

# クラスラベルを取得
y = iris.target

# 一意なクラスラベルを出力
print('Class labels:', np.unique(y))

# 訓練データとテストデータに分割
# 全体の３０％をテストデータに
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# P.51 分割の確認
# 上記'Class lavels'でyには[0, 1, 2]だけが格納されていることを確認した。
# 分割後の[0, 1, 2]の配分を見てみる。
print('Label counts in y:', np.bincount(y))
print('Label counts in y_train:', np.bincount(y_train))
print('Label counts in y_test:', np.bincount(y_test))


# P.51 特徴量を標準化
sc = StandardScaler()

# 訓練データの平均と標準偏差を計算
sc.fit(X_train)

# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(X_train)
print(X_train_std)
