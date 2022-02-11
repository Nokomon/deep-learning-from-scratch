from common.functions import *
from common.layers import *
from common.np import *
from common.config import GPU

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        dx_sum = np.sum(dx, axis=1, keepdims=True)   # why?
        dx = dx - self,out * dx_sum
        return dx

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
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0   # dW의 모든 element를 0으로

        """
        아래 코드는, 같은 단어에 대해 역전파할 때 값이 덮어쓰이는 문제 발생.
        따라서, 같은 단어에 대해 역전파한다면 그 값을 더해줘야 하고,
        이유는 matmul 계층을 역전파하는 과정에서 dW 값을 도출할 때 값들이 더해지기 때문
        따라서, 라인 76-77과 같이 진행.
        """
        # dw[self.idx] = dout
        if GPU:
            import cupyx
            cupyx.scatter_add(dW, self.idx, dout)
        else:
            # for i, word_id in enumerate(self.idx):
            #     dW[word_id] += dout[i]
            np.add.at(dW, self.idx, dout)
        return

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out   # 계층을 통과한 y값 저장 (역전파 때 사용)
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out   # 순전파 결과 사용
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None   # 출력
        self.t = None   # 정답

    def forward(self, x, t):
        # 정답/오답 여부: 각각 1 or 0으로
        # 후에 NegativeSamplingLoss 클래스에서 np.ones 또는 np.zeros로 label을 만들고
        # 이를 파라미터 t로 받음
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)   # np.c_: concatenate (세로로)_
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size   # dL/dx = y-t로부터
        return dx




