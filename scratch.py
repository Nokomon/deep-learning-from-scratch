import numpy as np

test = np.arange(120).reshape(2, 3, 4, 5)
print(test)

test[...] = 0
print(test)