# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:54:13 2021

@author: KOMATSU
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------
# 2-2-2_get_iris_data.py
# ----------------------
s = './data/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# ----------------------
# 2-3-1_create_Iris_graph1.py
# ----------------------
# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values

# Iris-setosaを-1, Iris-versicolorを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)

# 1-100行目の1,3列目の抽出
X = df.iloc[0:100, [0, 2]].values

# 品種 setosaのプロット（赤〇）
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# 品種 setosaのプロット（青〇）
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x',
            label='versicolor')

# 軸のラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 汎用の設定(左上に配置)
plt.legend(loc='upper left')

# 図の表示
plt.show()
