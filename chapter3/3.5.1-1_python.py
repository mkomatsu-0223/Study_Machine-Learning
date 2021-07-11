# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# 乱数シードを設定
np.random.seed(1)
# 標準正規分布に従う乱数で２００行２列の行列を生成
X_xor = np.random.randn(200, 2)
# ２つの引数に対して排他的論理和を実行
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
# 排他的論理和の値が真の場合は１，偽の場合は－１を割り当てる。
y_xor = np.where(y_xor, 1, -1)
# ラベル１を青のxでプロット
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x',
            label='1')
# ラベル―１を赤の四角でプロット
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s',
            label='-1')

# 軸ラベルの決定
plt.xlim([-3, 3])
plt.ylim([-3, 3])
# 凡例の設定（左上に配置）
plt.legend(loc='best')
# グラフを表示
plt.tight_layout()
plt.show()
