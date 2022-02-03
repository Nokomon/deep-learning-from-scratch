import numpy as np

N = 100
V = 650

total = N * V
print(total)

a = np.random.rand(N, V)
a = a > 0.5
a_sum = a.sum()
# print(a_sum)
print(a_sum / total * 100)

b = np.random.randn(N, V)
b = b > 0.5
b_sum = b.sum()
# print(b_sum)
print(b_sum / total * 100)

c = np.random.uniform(0, 1, (N, V))
c = c > 0.5
c_sum = c.sum()

print(c_sum / total * 100)