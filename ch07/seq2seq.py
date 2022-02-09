import sys
sys.path.append('..')

import numpy as np

from common.time_layers import *
from common.base_model import BaseModel

class Encoder:
    # Time Embedding -> Time LSTM -> returns hidden state(h) as a final output
    def __init__(self, vocab_size, vector_size, hidden_size):
        V, D, H = vocab_size, vector_size, hidden_size

        W_embed = (np.random.randn(V, D) / np.sqrt(V)).astype('f')
        Wx_lstm = (np.random.randn(D, 4*H) / np.sqrt(D)).astype('f')
        Wh_lstm = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        b_lstm = np.zeros(4*H).astype('f')

        self.time_embed = TimeEmbedding(W_embed)
        self.time_lstm = TimeLSTM(Wx_lstm, Wh_lstm, b_lstm, stateful=False)
        """
        - stateful = False로 두어, 문제마다 LSTM 은닉 상태 영행렬로 초기화
        - stateful = True로 한다면, 영행렬이 아니라 TimeLSTM에서 다음 TimeLSTM으로 h가 넘겨짐 -> 대참사
        """

        self.params = self.time_embed.params + self.time_lstm.params
        self.grads = self.time_embed.grads + self.time_lstm.grads

        self.hs = None

    def forward(self, xs):
        xs = self.time_embed.forward(xs)
        hs = self.time_lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]   # returns the last hidden state of the LSTM layer

    def backward(self, dh):   # dh: Decoder가 전해주는 gradient
          dhs = np.zeros_like(self.hs)
          dhs[:, -1, :] = dh

          dout = self.time_lstm.backward(dhs)
          dout = self.time_embed.backward(dout)
          return dout

class Decoder:
    # Time Embedding -> Time LSTM -> Time Affine
    def __init__(self, vocab_size, vector_size, hidden_size):
        V, D, H = vocab_size, vector_size, hidden_size

        W_embed = (np.random.randn(V, D) / np.sqrt(V)).astype('f')
        Wx_lstm = (np.random.randn(D, 4*H) / np.sqrt(D)).astype('f')
        Wh_lstm = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        b_lstm = np.zeros(4*H).astype('f')
        W_affine = (np.random.randn(H, V) / np.sqrt(H)).astype('f')
        b_affine = np.zeros(V).astype('f')

        self.time_embed = TimeEmbedding(W_embed)
        self.time_lstm = TimeLSTM(Wx_lstm, Wh_lstm, b_lstm, stateful=True)
        """
        - 이 경우에는, Encoder에서 h를 받아와야 하기 때문에 stateful=True로 지정해줌.
        - stateful=False일 경우, 받아오더라도 h를 영행렬로 초기화하여 진행 -> 대참사
        """
        self.time_affine = TimeAffine(W_affine, b_affine)

        self.params, self.grads = [], []
        for layer in [self.time_embed, self.time_lstm, self.time_affine]:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.time_lstm.set_state(h)   # Encoder의 맥락벡터 전달
        out = self.time_embed.forward(xs)
        out = self.time_lstm.forward(out)
        score = self.time_affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.time_affine.backward(dscore)
        dout = self.time_lstm.backward(dout)
        dout = self.time_embed.backward(dout)
        dh = self.time_lstm.dh   # Encoder로 전달해줄 기울기
        return dh

    # 학습: forward, 생성: generate -> teacher forcing 여부에 따라 다름
    def generate(self, h, start_id, sample_size):
        self.time_lstm.set_state(h)
        sampled = []
        sample_id = start_id

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.time_embed.forward(x)
            out = self.time_lstm.forward(out)
            score = self.time_affine.forward(out)

            # sampling: 숫자 데이터에서는 결정론적으로 샘플링한다.
            sample_id = np.argmax(score.flatten())   # for문을 돌면서 다시 input으로 들어감
            sampled.append(int(sample_id))
        return sampled

class Seq2seq(BaseModel):
    def __init__(self, vocab_size, vector_size, hidden_size):
        V, D, H = vocab_size, vector_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        h = self.encoder.forward(xs)

        decoder_input, decoder_label = ts[:, :-1], ts[:, 1:]
        score = self.decoder.forward(decoder_input, h)
        loss = self.softmax.forward(score, decoder_label)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled



