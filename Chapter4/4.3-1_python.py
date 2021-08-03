# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


# サンプルデータを読み込み
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

# 列名を取得
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

# クラスラベルを表示
print('Class labels', np.unique(df_wine['Class label']))

# 表示
df_wine.head()
df_wine.to_csv('df_wine.csv')
