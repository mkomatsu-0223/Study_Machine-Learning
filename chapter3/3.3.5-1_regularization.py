# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:54:13 2021

@author: KOMATSU
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# Irisデータをロード
iris = datasets.load_iris()

# 3、4列目の特徴量を抽出
X = iris.data[:, [2, 3]]

# クラスラベルを取得
y = iris.target

# 訓練データとテストデータに分割
# 全体の３０％をテストデータに
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# P.51 特徴量を標準化
sc = StandardScaler()

# 訓練データの平均と標準偏差を計算
sc.fit(X_train)

# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 訓練データとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# 訓練データとテストデータのクラスラベルを結合
y_combined_std = np.hstack((y_train, y_test))


# 空のリストを生成(重み係数、逆正則化パラメータ)
weights, params = [], []
# １０個の逆正則化パラメータに対するロジスティック回帰モデルをそれぞれ処理
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1, solver='lbfgs',
                            multi_class='ovr')
    lr.fit(X_train_std, y_train)
    # 重み係数を格納
    weights.append(lr.coef_[1])
    # 逆正則化パラメータを格納
    params.append(10.**c)

# 重み係数をNumpy配列に変換
weights = np.array(weights)

# 決定領域をプロット
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], label='petal width', linestyle='--')
# 軸ラベルを追加
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.legend(loc='upper left')
# 横軸を対数スケールに設定
plt.xscale('log')
plt.show()
