import sys
sys.path.append('..')

import numpy as np
from collections import Counter
from common.layers import *

# Embedding + dot product
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads

        # 순잔파 시의 계산 결과를 잠시 유지하기 위한 변수
        # 순전파 후 역전파 때 순전파때 사용된 맥락정보, W_out의 열벡터를 불러오기 위함
        self.cache = None

    def forward(self, h, idx):
        """
        h: 맥락 벡터 (은닉층 도달하기 전 contexts의 W_in 행벡터의 평균)
        idx: 정답/오답 인덱스
        """
        target_W = self.embed.forward(idx)   # idx: 배열 for 미니배치 처리. 쉽게 말해 W_out의 열벡터를 구하는 과정
        # out = np.matmul(h, target_W.T)   # target_W 열벡터와 h 행벡터의 내적
        out = np.sum(target_W*h, axis=1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)   # 2차원 행벡터로 -> shape 맞춰주기 위해
        dtarget_W = dout * h   # dot 계층의 역전파: 서로 바꾼다
        dh = dout * target_W   # dot 계층의 역전파: 서로 바꾼다
        self.embed.backward(dtarget_W)   # 역전파하여 Embedding 레이어의 dW 업데이트
        return dh   # 역전파를 계속하기 위해 dh 전달

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
                                               replace=False,
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
