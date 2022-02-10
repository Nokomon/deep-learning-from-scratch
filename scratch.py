import numpy as np

a = np.arange(10).reshape(1, 5, 2)
b = np.arange(2).reshape(1, 1, 2)

print(a)
print(b)

c = np.concatenate((a, b), axis=2)
print(c)