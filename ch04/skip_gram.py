import sys
sys.path.append('..')

from common.layers import *
from ch04.negative_sampling_layer import NegativeSamplingLoss

class SkipGram:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = (np.random.randn(V, H) / np.sqrt(V)).astype('f')
        W_out = (np.random.randn(V, H) / np.sqrt(V)).astype('f')

        # 계층 생성
        self.in_layer = Embedding(W_in)
        self.ns_loss_layers = []
        for i in range(2 * window_size):
            layer = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
            self.ns_loss_layers.append(layer)

        # 모든 가중치와 기울기를 하나로 모은다
        layers = [self.in_layer] + self.ns_loss_layers
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 임베딩 결과 저장
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        loss = 0
        for i, layer in enumerate(self.ns_loss_layers):
            loss += layer.forward(h, contexts[:, i])
        return loss

    def backward(self, dout=1):
        dh = 0
        for i, layer in enumerate(self.ns_loss_layers):
            dh += layer.backward(dout)
        self.in_layer.backward(dh)
        return