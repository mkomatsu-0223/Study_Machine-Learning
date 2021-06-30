# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:54:13 2021

@author: KOMATSU
"""

# import os
import pandas as pd


# ----------------------
# 2-2-2_get_iris_data.py
# ----------------------
# s = os.path.join('https://archive.ics.uci.edu', 'ml',
#                 'machine-learning-databases', 'iris', 'iris.data')
s = './data/iris.data'

df = pd.read_csv(s, header=None, encoding='utf-8')

print(df.tail(5))
