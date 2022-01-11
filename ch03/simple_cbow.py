import sys
sys.path.append('..')

import numpy as np
from common.layers import *

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = np.random.randn(V, H)
        W_out = np.random.randn(H, V)

        # 계층
        self.in_layer0 = Matmul(W_in)
        self.in_layer1 = Matmul(W_in)
        self.out_layer = Matmul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        layers = [self.in_layer0, self.in_layer1, self.out_layer]

        # 가중치, 기울기를 리스트에 모음
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # Embedding된 단어 벡터(W_in 행렬) 인스턴스 변수로 저장
        # 때에 따라서는 W_in, W_out, 혹은 둘다 사용
        self.word_vecs = W_in + W_out.T

    def forward(self, contexts, target):
       h0 = self.in_layer0.forward(contexts[:, 0])   # 미니배치, target 전 단어
       h1 = self.in_layer1.forward(contexts[:, 1])   # 미니배치, tatget 후 단어
       h = (h0+h1) / 2
       score = self.out_layer.forward(h)
       loss = self.loss_layer.forward(score, target)
       return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5   # x0.5 계층
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return
