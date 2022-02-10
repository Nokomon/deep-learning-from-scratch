import sys

import numpy as np

sys.path.append('..')

from common.time_layers import *
from seq2seq import Seq2seq, Encoder

# Encoder는 같은 것을 쓰고, Decoder만 'peeky' 기능을 추가해서 다시 구현

class PeekyDecoder:
    # PeekyDecoder는 맥락 벡터(h; or 맥락 행렬)를 LSTM과 Affine 계층에도 넣어준다
    # h의 shape는 Time을 고려하지 않는다면 (N, H)
    def __init__(self, vocab_size, vector_size, hidden_size):
        V, D, H = vocab_size, vector_size, hidden_size

        W_embed = (np.random.randn(V, D) / np.sqrt(V)).astype('f')
        Wx_lstm = (np.random.randn(H + D, 4*H) / np.sqrt(H + D)).astype('f')
        Wh_lstm = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        b_lstm = np.zeros(4*H).astype('f')
        W_affine = (np.random.randn(H + H, V) / np.sqrt(H + H)).astype('f')
        b_affine = np.zeros(V).astype('f')

        self.time_embed = TimeEmbedding(W_embed)
        self.time_lstm = TimeLSTM(Wx_lstm, Wh_lstm, b_lstm, stateful=True)
        self.time_affine = TimeAffine(W_affine, b_affine)

        self.params, self.grads = [], []
        for layer in [self.time_embed, self.time_lstm, self.time_affine]:
            self.params += layer.params
            self.grads += layer.grads

        self.cache = H

    def forward(self, xs, h):
        self.time_lstm.set_state(h)

        N, T = xs.shape
        _, H = h.shape

        # TimeEmbedding
        out = self.time_embed.forward(xs)
        hs = np.repeat(h, T, axis=0)   # h를 T번만큼 복제 -> shape: (N*T, H)
        hs = hs.reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)   # 모든 t별 LSTM에 h라는 정보를 주기 위함

        # TimeLSTM
        out = self.time_lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)   # 모든 t별 Affine에 h라는 정보를 주기 위함

        # TimeAffine
        score = self.time_affine.forward(out)

        return score

    def backward(self, dscore):
        H = self.cache
        dhs = 0   # back propagation of 'peeky' h(s) -> to be added with dh later

        # TimeAffine 역전파
        dout = self.time_affine.backward(dscore)   # dout shape: (N, T, D)
        dhs += dout[:, :, :H]
        dout = dout[:, :, H:]

        # TimeLSTM 역전파
        dout = self.time_lstm.backward(dout)
        dhs += dout[:, :, :H]   # update dhs
        dout = dout[:, :, H:]

        # TimeEmbedding 역전파
        self.time_embed.backward(dout)

        # Encoder에 전해줄 dh 구하기 (final process for Decoder back propogation)
        dhs = np.sum(dhs, axis=1)   # 분기 노드의 역전파
        dh = self.time_lstm.dh + dhs   # 분기 노드의 역전파

        return dh

    def generate(self, h, start_id, sample_size):
        self.time_lstm.set_state(h)

        sampled = []
        char_id = start_id

        H = self.cache
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))   # initial input

            # TimeEmbedding
            out = self.time_embed.forward(x)
            out = np.concatenate((peeky_h, out), axis=2)
            print(out.shape)

            # TimeLSTM
            out = self.time_lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)

            # TimeAffine
            score = self.time_affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled




class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, vector_size, hidden_size):
        # super().__init__(vocab_size, vector_size, hidden_size)
        V, D, H = vocab_size, vector_size, hidden_size

        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

