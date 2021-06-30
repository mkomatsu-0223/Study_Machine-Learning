# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:54:13 2021

@author: KOMATSU
"""
import numpy as np


# ----------------------
# 2-5-1_AdalineGD.py
# ----------------------
class AdalineGD(object):
    """ADAptive LInear NEuron分類機

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
    cost_ : リスト
        各エポックでの誤差平方和のコスト関数

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        # 学習率の初期化
        self.eta = eta
        # 訓練回数の初期化
        self.n_iter = n_iter
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
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):  # 訓練回数分まで訓練データを反復
            net_input = self.net_input(X)
            # activationメソッドは単なる恒等関数であるため、
            # このコードではなんの効果もないことに注意。代わりに
            # 直接 'output = self.activation(net_input(X))' と記述することもできた。
            # activationメソッドの目的は、より概念的なものである。
            # つまり、（後ほど説明する）ロジスティック回帰の場合は
            # ロジスティック回帰の分類器を実装するためにシグモイド関数に変更することもできる。
            output = self.activation(net_input)
            # 誤差･･･参考書P37参照
            errors = (y - output)
            # ･･･参考書P37参照
            # ･･･参考書P37参照
            self.w_[1:] += self.eta * X.T.dot(errors)
            # ･･･参考書P37参照
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算
            cost = (errors**2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)
        return self

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
