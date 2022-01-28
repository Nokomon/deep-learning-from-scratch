from common.functions import *
from common.layers import *
from common.np import *


# RNN cell 한 개
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(i) for i in self.params]
        self.cache = None  # 순전파 결과 저장

    # 순전파: 잘 생각해보면 한 셀에 들어가는 요소들은 input(x)과 전 셀의 은닉 생태(h_prev)
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        temp_h_next = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(temp_h_next)
        self.cache = (x, h_prev, h_next)

        # 한 셀에서 처리한 정보를 다음 셀로 넘긴다
        return h_next

    def backward(self, dh_next):  # dh_next: 바로 전 셀에서 가져온 역전파값
        x, h_prev, h_next = self.cache
        Wx, Wh, b = self.params

        dt = dh_next * (1 - h_next ** 2)  # tanh 역전파
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dx = np.matmul(dt, Wx.T)
        dWx = np.matmul(x.T, dt)

        # save gradients for future use
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev


# T개의 input(x_0 ~ x_T-1)을 받아 T개의 은닉 상태를 반환하는
# T개의 RNN 셀이 붙어있는 Time RNN 계층 구현
### Time 계층을 구현할 때는 항상 dimension 선언하면서 구현할 것 ###
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):  # stateful: 은닉 상태를 인계받을 것인가?
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None  # 후에 다수의 RNN 계층을 리스트로 저장하는 용도로 사용

        self.h = None  # forward 호출 시 마지막 RNN 셀의 hidden state
        self.dh = None  # backward 호출 시 바로 전 블록의 hidden state gradient(dh_prev)
        self.stateful = stateful

    # 확장성 고려: TimeRNN 게층의 은닉 상태를 설정
    def set_state(self, h):
        self.h = h

    # 확장성 고려: TimeRNN 게층의 은닉 상태를 None으로 초기화
    def reset_state(self):
        self.h = None

    def forward(self, xs):  # xs: T개 분량의 시계열 데이터를 하나로 모음
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # N: 미니배치 크기, T: T개분량 시계열 데이터, D: 입력벡터 차원수
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        # stateful 하지 않거나, 처음 호출 시
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')  # self.h를 영행렬로 초기화

        for t in range(T):
            layer = RNN(Wx, Wh, b)
            x = xs[:, t, :]
            self.h = layer.forward(x, self.h)
            hs[:, t, :] = self.h  # 각 t마다 은닉 상태 벡터(h) hs에 차곡차곡 저장
            self.layers.append(layer)  # 역전파 때 사용하기 위해 layer정보 append해서 저장
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.zeros((N, T, D), dtype=np.float32)  # 모든 t에 대한 dx를 담을 '그릇'
        dh = 0
        grads = [0, 0, 0]  # 각각 dWx, dWh, b의 합을 담을 '그릇'

        # 각 RNN 셀에 대해서
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)  # 순전파 때 분기 -> 역전파 때 더해줌
            dxs[:, t, :] = dx

            # 윗줄의 backward로 인해서 이미 gradient값이 모두 업데이트 되었을 것
            # 따라서, 현재 보고 있는 시각 t에 대해 업데이트된 gradient들을 불러와서,
            # 누적으로 계속해서 모든 t에 대해 더해준다
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        # 위에서 도출된 gradient값을 인스턴스 변수에 덮어쓴다
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        # 지금 당장 RNN에서는 쓰진 않는다. BPTT이기에 다음 블록에 역전파 결괏값 전달해 줄 필요 없다
        # 다만, 후에 Seq2seq 구현을 위해 이와 같이 저장함 (RNNLM에서는 안씀!)
        self.dh = dh
        # print(dh)

        return dxs


# Time Affine -> T개의 Affine게층 한번에 "하는듯하게"
# 실제로는 T개의 Affine 꼐층을 사용하지 않고, "그럴듯하게" 함 -> reshape통해서 한번에 행렬곱
class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(i) for i in self.params]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape  # 실제 구현(RNNLM)에 가서는 N, T, H
        W, b = self.params

        """
        [효율이 좋은 이유에 대한 분석]
        - (N, T, D)의 형상으로 한다는 것의 의미는 곧 (N, D)의 Affine 계층 input을 T번 호출한다는 의미
        - 이것보다는, (NxT, D)의 형상으로 가져감으로써,  T번 호출할 필요 없이 한 번의 행렬곱으로 처리가능
        - (NxT, D)의 의미를 생각해보자.
            - N은 미니배치 크기, D는 임베딩 차원, T는 Time계층에서 몇 개의 RNN cell을 가져올 것인가의 의미
            - 그러므로, (NxT, D)라고 하면 '모든'(N, T 모두 고려한 '모든') input을 하나로 모은 것
            - 즉, ndim=2의 형태에서, 모든 input을 불러와서, 각 행은 embedded word vector임을 의미  
        """
        x_reshape = x.reshape(N * T, -1)
        out = np.dot(x_reshape, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        x_reshape = x.reshape(N * T, -1)

        dW = np.dot(x_reshape.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(N, T, D)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None  # 순전파 후 역전파 처음 시작할 때 순전파 정보 불러오기 위함

    def forward(self, xs, ts):
        N, T, V = xs.shape
        if ts.ndim == 3:  # 정답 레이블이 원핫 벡터인 경우
            ts = ts.argmax(axis=2)  # 1인 인덱스만 뽑아서 ts에 넣는다 -> 형상: (N, t)
        mask = (ts >= 0)  # 불린값. 배열로 반환되어, ts와 같은 값인 것만 False. ignore_label 무시하여 작성

        # 배치용과 시계열용을 정리 (reshape) for "효율적인" 계산
        xs = xs.reshape(N * T, V)  # 3d -> 2d
        ts = ts.reshape(N * T)  # 2d -> 1d
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])  # 교차 엔트로피(L) 결괏값들

        # True를 곱해주면 그대로, False를 곱해주면 0으로
        # 즉, ignore_label에 해당하는 데이터는 손실을 0으로 설정한다.
        ls *= mask
        loss = -np.sum(ls)
        loss /= mask.sum()  # True인 요소들의 개수로 나눈다

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    # SoftmaxWithLoss의 역전파 보고 다시 볼 것
    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        dx = dx.reshape((N, T, V))
        return dx


class TimeEmbedding:
    def __init__(self, W):
        self.W = W
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None

    def forward(self, xs):
        N, T = xs.shape
        H, D = self.W.shape

        result = np.zeros((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):  # RNN이 아니기 때문에 reverse 해줄필요 없다. 이건 Embedding 모아놓은 것일뿐
            layer = Embedding(self.W)
            self.layers.append(layer)
            result[:, t, :] = layer.forward(xs[:, t])

        return result

    def backward(self, dout):
        N, T, D = dout.shape  # TimeRNN 역전파 결과

        gradient = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            gradient += layer.grads[0]  # dW

        self.grads[0][...] = gradient
        return


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(i) for i in self.params]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        # 1. 항상 params 불러오고, 차원 정의
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # 2. 아핀 변환, 슬라이싱, 그리고 input으로 넣을 준비
        affined = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        f, g, i, o = affined[:, :H], affined[:, H:2*H], affined[:, 2*H:3*H], affined[:, 3*H:]
        f, g, i, o = sigmoid(f), np.tanh(g), sigmoid(i), sigmoid(o)

        # 3. LSTM 본격 구현
        c_next = (c_prev * f) + (g * i)
        h_next = np.tanh(c_next) * o
        self.cache = x, h_prev, c_prev, f, g, i, o, c_next
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, f, g, i, o, c_next = self.cache

        do = dh_next * np.tanh(c_next)
        dc_ongoing = dc_next + (1 - np.tanh(c_next) ** 2) * (dh_next * o)
        dc_prev = f * dc_ongoing
        di = g * dc_ongoing
        dg = i * dc_ongoing
        df = c_prev * dc_ongoing

        daffined = np.hstack((df, dg, di, do))
        db = daffined.sum(axis=0)

        dWx = np.matmul(x.T, daffined)
        dx = np.matmul(daffined, Wx.T)

        dWh = np.matmul(h_prev.T, daffined)
        dh_prev = np.matmul(daffined, Wh.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev, dc_prev

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(i) for i in self.params]
        self.stateful = stateful
        self.layers = None
        """
        forward 메소드를 계속해서 BetterRnnlm(궁극적으론 RnnlmTrainer)에서 호출하게 되는데,
        이때, self.layers에 LSTM 계층 한 개를 계속해서 append해준다.
        그러나, self.layers = []로 하게 된다면, 학습이 끝날때까지 리스트에 계속해서 append해주기 때문에
        빈 리스트가 아닌, None 등으로 초기화시켜 주는 것
        """

        self.dh = None
        self.h, self.c = None, None

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        hs = np.zeros((N, T, H), dtype='f')
        self.layers = []
        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            # print(f"Forward{t}: {len(self.layers)}")
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            self.layers.append(layer)
            hs[:, t, :] = self.h

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.zeros((N, T, D), dtype='f')
        dh, dc = 0, 0
        grads = [0 for _ in range(len(self.grads))]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)  # 순전파 때 분기 -> 역전파 때 더해줌
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs


class TimeDropout:
    def __init__(self, ratio=0.5):
        self.params, self.grads = [], []
        self.ratio = ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.randn(*xs.shape) > self.ratio
            scale = 1 / (1.0 - self.ratio)
            self.mask = flg.astype(np.float32) * scale
            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask

