import numpy as np


a = np.arange(10).reshape(2, 5)
b = np.arange(10).reshape(5, 2)

r1 = np.matmul(a, b)
r2 = np.dot(a, b)

print(r1)
print(r2)
print(r1 == r2)