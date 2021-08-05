# -*- coding: utf-8 -*-
import numpy as np

ex = np.array([0, 1, 2, 3, 4, 5])

standardized = (ex - ex.mean()) / ex.std()
print('standardized:', standardized)

normalized = (ex - ex.min()) / (ex.max() - ex.min())
print('normalized:', normalized)
