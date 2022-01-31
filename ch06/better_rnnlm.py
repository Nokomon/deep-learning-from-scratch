"""
<BetterRnnlm의 개선점>
    1. LSTM 계층의 다층화(여기서는 2층으로)
    2. Dropout 추가 (깊이 반향으로만)
        cf. "Variational Dropout": 같은 계층끼리 mask 공유 -> 시간 방향으로도 가능
    3. 가중치 공유 (Embedding 계층과 Affine 계층)
"""

import sys
sys.path.append('..')

from common.time_layers import *
from common.np import *
from common.base_model import BaseModel

class BetterRnnlm(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio):
        V, D, H = vocab_size, wordvec_size, hidden_size

        ### Initialize respective layer parameters ###
        # TimeEmbedding
        W_embed = (np.random.randn(V, D) / np.sqrt(V)).astype('f')

        # LSTM 1
        Wx_lstm1 = (np.random.randn(D, 4*H) / np.sqrt(D)).astype('f')
        Wh_lstm1 = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        b_lstm1 = np.zeros(4*H).astype('f')

        # LSTM 2: 2층 LSTM의 입력은 1층 LSTM의 은닉 상태임을 고려하여 차원 구성
        Wx_lstm2 = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        Wh_lstm2 = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        b_lstm2 = np.zeros(4*H).astype('f')

        # Time Affine: 가중치 공유를 고려하여 편향만 초기화
        b_affine = np.zeros(V).astype('f')


        # Layers
        self.layers = [
            TimeEmbedding(W_embed),
            TimeDropout(dropout_ratio),
            TimeLSTM(Wx_lstm1, Wh_lstm1, b_lstm1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(Wx_lstm2, Wh_lstm2, b_lstm2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(W_embed.T, b_affine)   # 가중치 공유
        ]

        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        prediction = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(prediction, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()