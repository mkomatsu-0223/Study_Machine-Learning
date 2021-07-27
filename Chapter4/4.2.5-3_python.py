# -*- coding: utf-8 -*-
import pandas as pd


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

# One-hotエンコーディングを実行
df_temp = pd.get_dummies(df[['price', 'color', 'size']])

print(type(df_temp))
print(df_temp)
