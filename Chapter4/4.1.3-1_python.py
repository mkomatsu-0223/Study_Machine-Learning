# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer


# サンプルデータを作成
csv_data = '''A,B,C,D
                1.0,2.0,3.0,4.0
                5.0,6.0,,8.0
                10.0,11.0,12.0,
                '''

# サンプルデータを読み込み
df = pd.read_csv(StringIO(csv_data))

# 欠測値補完のインスタンスを生成（平均値補完）
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# データを適合
imr = imr.fit(df.values)

# 補完を実行
imputed_data = imr.transform(df.values)
print('---------')
print(imputed_data)

print('---------')
print(df)
print('---------')
print(df.mean())
print(df.fillna(df.mean()))
