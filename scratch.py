import numpy as np

a = np.random.rand(2, 5)
print(a > 0.5)
a = a > 0.5
print(a.sum())

b = np.random.randn(2, 5)
b = b > 0.5
print(b.sum())