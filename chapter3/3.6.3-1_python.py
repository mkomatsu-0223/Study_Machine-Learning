# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # 各特徴量を１次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # クラスごとに訓練データをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')
    # テストデータ点を目立たせる(点を○で表示)
    if test_idx:
        # すべてのデータ点をプロット
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='y', edgecolors='black',
                    alpha=0.3, linewidths=1, marker='o', s=100,
                    label='test set')


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

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))


# ジニ不純度を指数とするランダムフォレストのインスタンスを生成
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, random_state=1, n_jobs=2)

# ランダムフォレストのモデルを訓練データに適合させる
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest,
                      test_idx=range(105, 150))
# 軸ラベルを追加
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
# 凡例の設定（左上に配置）
plt.legend(loc='upper left')
# グラフを表示
plt.tight_layout()
plt.show()
