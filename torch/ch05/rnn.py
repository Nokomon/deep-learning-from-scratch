import torch
import torch.nn as nn

f = torch.float

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [torch.zeros_like(i) for i in self.params]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        element1 = torch.matmul(h_prev, Wh)
        element2 = torch.matmul(x, Wx)
        h_next = torch.tanh(element1 + element2 + b)
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        x, h_prev, h_next = self.cache
        dt = dh_next(1 - h_next**2)
        db = torch.sum(dt, dim=0)
        dWh = torch.matmul(h_prev.T, dt)
        dh_prev = torch.matmul(dt, Wh.T)
        dx = torch.matmul(dt, Wx.T)
        dWx = torch.matmul(x.T, dt)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=True):
        self.params = [Wx, Wh, b]
        self.grads = [torch.zeros_like(i) for i in self.params]
        self.h = None
        self.layers = None
        self.stateful = stateful

    def set_state(self, h, reset=False):
        if reset:
            self.h = None
        else:
            self.h = h

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        _, H = Wx.shape

        self.layers = []
        hs = torch.zeros((N, T, D), dtype=f)

        if not self.stateful or self.h is None:
            self.h = torch.zeros((N, H), dtype=f)

        for t in range(T):
            layer = RNN(Wx, Wh, b)
            self.layers.append(layer)
            x, h = xs[:, t, :], self.h
            hs[:, t, :] = layer.forward(x, h)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, _ = Wx.shape

        dxs = torch.zeros((N, T, D), dtype=f)
        dh = 0
        grads = [0 for _ in range(len(self.params))]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            grads[i][...] = grad

        return dxs

class TimeEmbedding:
    def __init__(self, W):
        self.W = W
        self.params = [W]
        self.grads = [torch.zeros_like(W)]
        # self.layers = None

    def forward(self, xs):
        N, T, D = xs.shape
        H, _ = self.W.shape

        self.layers = []
        for t in range(T):
            layer = nn.Embedding(num_embeddings=)