# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:54:13 2021

@author: KOMATSU
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------
# 2-2-1_Perceptron.py
# ----------------------
class Perceptron(object):
    """パーセプトロンの分類機

    パラメータ
    ----------
    eta : float
        学習率（0.0 より大きく1.0 以下の値）
    n_iter : int
        訓練データの訓練回数
    random_state : int
        重みを初期化するための乱数シード

    属性
    ----------
    w_ : １次元配列
        適合後の重み
    errors_ : リスト
        各エポックでの誤分類（更新）の数

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        パラメータ
        ----------
        X : {配列のようなデータ構造}, shape = [n_examples, n_features]
            訓練データ
            n_examplesは訓練データの個数、n_featuresは特徴量の個数
        y : 配列のようなデータ構造, shape = [n_examples]
            目的変数

        戻り値
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):  # 訓練回数分まで訓練データを反復
            errors = 0
            for xi, target in zip(X, y):  # 各訓練データで重みを更新
                # 重み w1, ... ., wmの更新
                # 参考書P25見て
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重み w0の更新：△w0 = n(yi - y^i)
                self.w_[0] += update
                # 重みの更新が 0 でない場合は誤分類としてカウント
                errors += int(update != 0.0)
            # 反復ごとの誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """ 総入力を計算 """
        # np.dotはベクトルのドット積 wTxを計算する。
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ １ステップ後のクラスラベルを返す """
        # numpy.where(condition[, x, y])はconditionを満たすときはxを、満たさない場合はyを返す。
        return np.where(self.net_input(X) >= 0.0, 1, -1)


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

# ----------------------
# 2-3-1_create_Iris_graph2.py
# ----------------------
# パーセプトロンのオブジェクトの生成（インスタンス化）
ppn = Perceptron(eta=0.1, n_iter=10)

# 訓練データへのモデルの適合
ppn.fit(X, y)

# エポックと誤分類の関係を表す折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Number of update')

# 図の表示
plt.show()
