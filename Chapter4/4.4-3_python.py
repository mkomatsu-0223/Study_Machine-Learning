# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split


# サンプルデータを読み込み
df_wine = pd.read_csv('df_wine.csv')

# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# 訓練データとテストデータに分割（全体の３０％をテストデータにする）
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# 標準化のインスタンスを生成（平均＝０、標準偏差＝１に変換）
stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
