# -*- coding: utf-8 -*-
import pandas as pd
from io import StringIO

# サンプルデータを作成
csv_data = '''A,B,C,D
                1.0,2.0,3.0,4.0
                5.0,6.0,,8.0
                10.0,11.0,12.0,
                '''

# サンプルデータを読み込み
df = pd.read_csv(StringIO(csv_data))

# 欠測値を含む行を削除
temp_df = df
print('---------------')
print(temp_df.dropna())

# 欠測値を含む列を削除
temp_df = df
print('---------------')
print(temp_df.dropna(axis=1))

# すべての列がNaNである行だけを削除。
# （今回はすべての値がNaNの行はないので配列全体が返される）
temp_df = df
print('---------------')
print(temp_df.dropna(how='all'))

# 非NaN値が４つ未満の行を削除
temp_df = df
print('---------------')
print(temp_df.dropna(thresh=4))

# 特定の列にNaNが含まれている行だけを削除
# （今回は'C'列にNaNが含まれている行）
temp_df = df
print('---------------')
print(temp_df.dropna(subset=['C']))
