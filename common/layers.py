import sys
sys.path.append('..')

from collections import Counter

from common.functions import *
from common.layers import *
from common.np import *
from common import config

# GPU Test
config.GPU = True
GPU = config.GPU

import numpy as np

class Matmul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1   # 정답 레이블이 1이어서 그런가
        dx *= dout
        dx = dx / batch_size
        return dx

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx   # 하나의 idx거나 multiple indices를 모은 array
        return W[idx]

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0   # dW의 모든 element를 0으로

        """
        아래 코드는, 같은 단어에 대해 역전파할 때 값이 덮어쓰이는 문제 발생.
        따라서, 같은 단어에 대해 역전파한다면 그 값을 더해줘야 하고,
        이유는 matmul 계층을 역전파하는 과정에서 dW 값을 도출할 때 값들이 더해지기 때문
        따라서, 라인 76-77과 같이 진행.
        """
        # dw[self.idx] = dout
        if GPU:
            import cupyx
            cupyx.scatter_add(dW, self.idx, dout)
        else:
            for i, word_id in enumerate(self.idx):
                dW[word_id] += dout[i]
        return

class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out   # 계층을 통과한 y값 저장 (역전파 때 사용)
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out   # 순전파 결과 사용
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None   # 출력
        self.t = None   # 정답

    def forward(self, x, t):
        # 정답/오답 여부: 각각 1 or 0으로
        # 후에 NegativeSamplingLoss 클래스에서 np.ones 또는 np.zeros로 label을 만들고
        # 이를 파라미터 t로 받음
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)   # np.c_: concatenate (세로로)_
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size   # dL/dx = y-t로부터
        return dx


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size   # 몇 개를 샘플링 할 것인지?
        self.vocab_size = None
        self.word_p = None   # 궁극적으로는 단어당 출현 확률분포

        counts = Counter()
        for word_id in corpus:
            counts[word_id] += 1
        # 여기까지: counts에는 "단어id: 빈도수" 저장

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
        # 여기까지: self.word_p에는 "단어: 빈도수" 저장

        self.word_p = np.power(self.word_p, power)   # Negative Sampling: 0.75승
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        """
        - target: 긍정적 예의 타깃의 모음. ndim=1인 배열로, 각 element는 긍정적 예의 word id
        """
        batch_size = target.shape[0]
        if not GPU:
            # 미니배치 크기 x 샘플 크기 형태로 영행렬 만듦
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0   # 자기 자신은 긍정적 예이기에, 뽑히면 안됨 -> 0으로 초기화
                p /= p.sum()   # 0으로 초기화된 element 제외하고 다시 확률 계산
                negative_sample[i, :] = np.random.choice(self.vocab_size,
                                                         size=self.sample_size,
                                                         replace=False,
                                                         p=p)
        else:
            # 이 경우에는 부정적 예에 타깃이 포함될 수 있다.
            # 긍정적 예인 자기 자신도 포함될 수 있다.
            # 실험할 때 수정?
            negative_sample = np.random.choice(self.vocab_size,
                                               size=(batch_size, self.sample_size),
                                               replace=True,
                                               p=self.word_p)
        return negative_sample

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        # sample_size+1개 만들어준다.
        # 부정적 예를 다루는 계층 sample_size개, 긍정적 예를 다루는 계층 1개
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]

        # 후에 W_out으로 받음, 가중치 공유
        # EmbeddingDot도 구현한 Embedding 기반 -> W_out도 W_in처럼 각 행이 단어 벡터 표현
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)   # 각각 x, t

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]   # 열 가져옴
            score = self.embed_dot_layers[i+1].forward(h, negative_target)
            loss += self.loss_layers[i+1].forward(score, negative_label)
        return loss

    def backward(self, dout=1):
        dh = 0   # 최종 결괏값 초기화
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh