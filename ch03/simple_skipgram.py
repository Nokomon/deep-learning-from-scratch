import sys
sys.path.append('..')

import numpy as np
from common.layers import *

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = np.random.randn(V, H)
        W_out = np.random.randn(H, V)

        # 계층
        self.in_layer = Matmul(W_in)
        self.out_layer0 = Matmul(W_out)
        self.out_layer1 = Matmul(W_out)
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()
        layers = [self.in_layer,
                  self.out_layer0,
                  self.out_layer1]

        # 가중치, 기울기를 리스트에 모음
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # W_in or W_out.T -> what to use for word embedding
        self.word_vecs = W_in

    def forward(self, contexts, target):
        # print(context)
        # print(context.shape)
        # print(targets)
        # print(targets.shape)
        h = self.in_layer.forward(contexts)
        o1 = self.out_layer0.forward(h)
        o2 = self.out_layer1.forward(h)
        loss1 = self.loss_layer0.forward(o1, target[:, 0])
        loss2 = self.loss_layer1.forward(o2, target[:, 1])
        loss = loss1 + loss2
        return loss

    def backward(self, dout=1):
        ds0 = self.loss_layer0.backward(dout)
        ds1 = self.loss_layer1.backward(dout)
        dm0 = self.out_layer0.backward(ds0)
        dm1 = self.out_layer1.backward(ds1)
        d_matmul = dm0 + dm1
        self.in_layer.backward(d_matmul)
        return