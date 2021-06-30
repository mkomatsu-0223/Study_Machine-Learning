# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:54:13 2021

@author: KOMATSU
"""
import numpy as np
from numpy.random import seed


# ----------------------
# 2-5-2_AdalineSGD.py
# ----------------------
class AdalineSGD(object):
    """ADAptive LInear NEuron分類機

    パラメータ
    ----------
    eta : float
        学習率（0.0 より大きく1.0 以下の値）
    n_iter : int
        訓練データの訓練回数
    shuffle : bool (デフォルト : True)
        Trueの場合は、循環を回避するためにエポックごとに訓練データをシャッフル
    random_state : int
        重みを初期化するための乱数シード

    属性
    ----------
    w_ : １次元配列
        適合後の重み
    cost_ : リスト
        各エポックでの誤差平方和のコスト関数

    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        # 学習率の初期化
        self.eta = eta
        # 訓練回数の初期化
        self.n_iter = n_iter
        # 重みの初期化フラグはFalseに設定
        self.w_initialized = False
        # 各エポックで訓練データをシャッフルするかどうかのフラグを初期化
        self.shuffle = shuffle
        # 乱数シードを設定
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
        # 重みベクトルの生成
        self._initialize_weights(X.shape[1])
        # コストを格納するリストの作成
        self.cost_ = []
        # 訓練回数分まで訓練データを反復
        for i in range(self.n_iter):
            # 指定された場合は訓練データをシャッフル
            if self.shuffle:
                X, y = self._shuffle(X, y)
            # 各訓練データのコストを格納するリストの生成
            cost = []
            # 各訓練データに対する計算
            for xi, target in zip(X, y):
                # 特徴量xiと目的変数yを用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi, target))
            # 訓練データの平均コストの計算
            avg_cost = sum(cost) / len(y)
            # 平均コストの格納
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """ 重みを再初期化することなく訓練データに適合させる """
        # 初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # 目的変数yの要素数が２以上の場合は各訓練データの特徴量xiと目的変数targetで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        # 目的変数yの要素数が１の場合は訓練データ全体の特徴量Xと目的変数yで重みを更新
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """ 訓練データをシャッフル """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """ 重みを小さな乱数に初期化 """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ ADALINEの学習規則を用いて重みを更新 """
        # 活性化関数の出力の計算
        output = self.activation(self.net_input(xi))
        # 誤差の計算
        error = (target - output)
        # 重み w1,･･･,wm の更新
        self.w_[1:] += self.eta * xi.dot(error)
        # 重み w0 の更新
        self.w_[0] += self.eta * error
        # コストの計算
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """ 総入力を計算 """
        # np.dotはベクトルのドット積 wTxを計算する。
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ 線形活性化関数の出力を計算 """
        return X

    def predict(self, X):
        """ １ステップ後のクラスラベルを返す """
        # numpy.where(condition[, x, y])はconditionを満たすときはxを、満たさない場合はyを返す。
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
