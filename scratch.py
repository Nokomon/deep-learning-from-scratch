import numpy as np


random = np.random.randn(30, 30)
# print(random)
flg = random > 0.5
# print(flg)
print(flg.sum())

# print(random[flg])
print(len(random[flg]))