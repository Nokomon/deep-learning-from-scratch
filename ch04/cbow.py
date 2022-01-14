import sys
sys.path.append('..')

import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = np.random.randn(V, H).astype('f')
        W_out = np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self,in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치의 기울기를 하나로
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 단어 임베딩 저장 -> W_in or W_out or both
        # NegativeSamplingLoss가 구현한 Embedding을 기반으로 하기에,
        # SimpleCBOW와 다르게 W_out도 각 행이 단어 벡터
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])   # 배치처리(열벡터 모음)
        h *= 1 / len(self.in_layers)   # 평균을 위해
        loss = self.ns_loss.forward(h, target)   # NegativeSampling
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)

        # 역전파 과정에서 평균내는 계층(1/N)의 역전파이기에
        # 1/N 처리를 해준다.
        dout *= 1 / len(self.in_layers)

        for layer in self.in_layers:
            layer.backward(dout)
        return
