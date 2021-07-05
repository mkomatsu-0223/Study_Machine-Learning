# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:54:13 2021

@author: KOMATSU
"""
import numpy as np
import matplotlib.pyplot as plt


# シグモイド関数を定義
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# y=1のコストを計算する関数
def cost_1(z):
    return - np.log(sigmoid(z))


# y=0のコストを計算する関数
def cost_0(z):
    return - np.log(1 - sigmoid(z))


# 0.1間隔で-7以上7未満のデータを生成
z = np.arange(-7, 7, 0.1)
# 生成したデータでシグモイド関数を実行
phi_z = sigmoid(z)

# 元のデータとシグモイド関数出力をプロット
plt.plot(z, phi_z)
# 垂直線を追加
plt.axvline(0.0, color='k')
# y軸の上限/下限を設定
plt.ylim(-0.1, 1.1)
# 軸のラベルを設定
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
# y軸の目盛を追加
plt.yticks([0.0, 0.5, 1.0])
# Axesクラスのオブジェクトの取得
ax = plt.gca()
# y軸の目盛に合わせて水平グリッド線を追加
ax.yaxis.grid(True)
# グラフを表示
plt.tight_layout()
plt.show()


# 0.1間隔で-10以上10未満のデータを生成
z = np.arange(-10, 10, 0.1)
# シグモイド関数を実行
phi_z = sigmoid(z)

# y=1のコスト計算関数を実行
c1 = [cost_1(x) for x in z]
# 元のデータとシグモイド関数出力をプロット
plt.plot(phi_z, c1, label='J(w) if y=1')

# y=0のコスト計算関数を実行
c0 = [cost_0(x) for x in z]
# 元のデータとシグモイド関数出力をプロット
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

# y軸の上限/下限を設定
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
# 軸のラベルを設定
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
# 凡例を設定
plt.legend(loc='upper center')
# グラフを表示
plt.tight_layout()
plt.show()
