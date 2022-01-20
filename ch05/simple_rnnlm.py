import sys
sys.path.append("..")

import numpy as np
from common.time_layers import *

class SimpleRnnlm:
    def __init__(self, vocab_size, hidden_size, vec_size):
        V, D, H = vocab_size, vec_size, hidden_size

        # 가중치 초기화
        # 필요한 params: TimeEmbedding(W_embed), TimeRNN(Wx, Wh, Wb), TimeAffine(W_affine, b_affine)
        # 여기서는 사비에르 초기화 안함
        W_embed = (np.random.randn(V, D) / 100).astype('f')
        Wx_rnn = (np.random.randn(D, H) / np.sqrt(D)).astype('f')
        Wh_rnn = (np.random.randn(H, H) / np.sqrt(H)).astype('f')
        b_rnn = np.zeros(H).astype('f')
        W_affine = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
        b_affine = np.zeros(V).astype('f')

        # 레이어 생성
        self.layers = [
            TimeEmbedding(W_embed),
            TimeRNN(Wx_rnn, Wh_rnn, b_rnn, stateful=True),
            TimeAffine(W_affine, b_affine)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # 모든 가중치와 기울기를 하나로
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()   # TimeRNN에서 self.h = None으로

