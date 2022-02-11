import sys
sys.path.append('..')

from common.np import *
from common.layers import Softmax

# 이거 나중에 softmax function으로 구현 다시 해보자
class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        # hs: Encoder TimeLSTM들의 은닉 상태, h: Decoder 첫 번째 STM의 은닉 상태
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H)
        hr = hr.repeat(T, axis=1)

        t = hs * hr

        score = np.sum(t, axis=2)
        a = self.softmax.forward(score)
        return a

    def backward(self, da):
        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1)
        dt = dt.repeat(H, axis=2)
        dhs = dt * hr
        dhr = hs * dt
        h = np.sum(dhr, axis=1)
        return dhs, dh


        