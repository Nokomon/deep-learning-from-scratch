
import numpy as np

class A:
    def __init__(self, W):
        self.W = W
        self.layers = []

    def f(self, x):
        print(self.W.shape)
        # self.layers = []
        for i in range(3):
            self.layers.append(i)
            # print(self.layers)
        return self.layers


array = np.array([1, 2, 3])
a = A(array)
a.f(3)
print(a.layers)