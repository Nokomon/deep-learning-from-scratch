import numpy as np

# RNN cell 한 개
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), no.zeros_like(b)]
        self.cache = None   # 순전파 결과 저장

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        temp_h_next = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(temp_h_next)
        self.cache = (x, h_prev, h_next)

        # 한 셀에서 처리한 정보를 다음 셀로 넘긴다
        return h_next

    def backward(self, dh_next):   # dh_next: 바로 전 셀에서 가져온 역전파값
        dt = dh_next * (1 - h_next**2)   # tanh 역전파
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np,matmul(dt, Wh.T)
        dx = np.matmul(dt, Wx.T)
        dWx = np.matmul(x.T, dt)

        # save gradients for future use
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return

# T개의 input(x_0 ~ x_T-1)을 받아 T개의 은닉 상태를 반환하는
# T개의 RNN 셀이 붙어있는 Time RNN 계층 구현
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):   # stateful: 은닉 상태를 인계받을 것인가?
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None   # 후에 다수의 RNN 계층을 리스트로 저장하는 용도로 사용

        self.h = None   # forward 호출 시 마지막 RNN 셀의 hidden state
        self.dh = None   # backward 호출 시 바로 전 블록의 hidden state gradient(dh_prev)
        self.stateful = stateful

    # 확장성 고려: TimeRNN 게층의 은닉 상태를 설정
    def set_state(self, h):
        self.h = h

    # 확장성 고려: TimeRNN 게층의 은닉 상태를 None으로 초기화
    def reset_state(self):
        self.h = None

    def forward(self, xs):   # xs: T개 분량의 시계열 데이터를 하나로 모음
        Wx, Wh, b = self.params
        N, T, D = xs.shape   # N: 미니배치 크기, T: T개분량 시계열 데이터, D: 입력벡터 차원수
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        # stateful 하지 않거나, 처음 호출 시
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')   # self.h를 영행렬로 초기화

        for t in range(T):
            layer = RNN(Wx, Wh, b)
            x = xs[:, t, :]
            self.h = layer.forward(x, self.h)
            hs[:, t, :] = self.h   # 각 t마다 은닉 상태 벡터(h) hs에 차곡차곡 저장
            self.layers.append(layer)
        return hs

    def backward(self):

