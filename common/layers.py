import sys
sys.path.append('..')

import numpy as np
from common.functions import *

class Matmul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1   # 정답 레이블이 1이어서 그런가
        dx *= dout
        dx = dx / batch_size
        return dx

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx   # 하나의 idx거나 multiple indices를 모은 array
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0   # dW의 모든 element를 0으로

        """
        아래 코드는, 같은 단어에 대해 역전파할 때 값이 덮어쓰이는 문제 발생.
        따라서, 같은 단어에 대해 역전파한다면 그 값을 더해줘야 하고,
        이유는 matmul 계층을 역전파하는 과정에서 dW 값을 도출할 때 값들이 더해지기 때문
        """
        # dw[self.idx] = dout

        """따라서, 아래 코드로 진행해야 한다."""
        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        return

# W_in = np.random.randn(6, 4)
# layer = Matmul(W_in)
# x = np.array([1] + [0] * 5)
#
# result = layer.forward(x)
# print(result.shape)
# print(W_in)
# print(result)
