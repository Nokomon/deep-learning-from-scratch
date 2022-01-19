import numpy as np

ts = np.arange(1, 6)
mask = (ts != 3)
print(type(mask))
print(mask)
mask = mask.reshape(1, -1)
print(mask)
print(mask.sum())


a = np.array([True, False, False, False, True])
print(a.sum())

for i in a:
    print(i)