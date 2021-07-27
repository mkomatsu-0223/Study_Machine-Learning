# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# サンプルデータを生成（Tシャツの色、サイズ、価格、クラスラベル）
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
    ])

# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
# 表示
print(df)
print('---------')

# Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
inv_size_mapping = {v: k for k, v in size_mapping.items()}

# ディクショナリを適用
df['size'] = df['size'].map(size_mapping)
# 適用したディクショナリを戻すときに有効化
# df['size'] = df['size'].map(inv_size_mapping)
# 表示
print(df)
print('---------')

# クラスラベルと整数を対応させるディクショナリを生成
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}

# ディクショナリを適用
df['classlabel'] = df['classlabel'].map(class_mapping)

# 適用したディクショナリを戻す
inv_class_mapping = {v: k for k, v in class_mapping.items()}
# ディクショナリを適用
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# 表示
print(df)
print('---------')

# ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print('---------')

# クラスラベルをもとに戻す
y = class_le.inverse_transform(y)
print(y)
print('---------')
