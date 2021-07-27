# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# サンプルデータを生成（Tシャツの色、サイズ、価格、クラスラベル）
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
    ])

# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']

# Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# ディクショナリを適用
df['size'] = df['size'].map(size_mapping)

# クラスラベルと整数を対応させるディクショナリを生成
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# ディクショナリを適用
df['classlabel'] = df['classlabel'].map(class_mapping)

# Ｔシャツの色、サイズ、価格を抽出
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]),
                              ('nothing', 'passthrough', [1, 2])])

X_temp = c_transf.fit_transform(X).astype(float)
# 表示
print(X_temp)
print('---------')
