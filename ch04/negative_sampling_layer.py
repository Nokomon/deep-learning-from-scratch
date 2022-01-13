import sys
sys.path.append('..')

import numpy as np
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
        out = np.matmul(h, target_W.T)   # target_W 열벡터와 h 행벡터의 내적
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)   # 2차원 행벡터로 -> shape 맞춰주기 위해
        dtarget_W = dout * h   # dot 계층의 역전파: 서로 바꾼다
        dh = dout * target_W   # dot 계층의 역전파: 서로 바꾼다
        self.embed.backward(dtarget_W)   # 역전파하여 Embedding 레이어의 dW 업데이트
        return dh   # 역전파를 계속하기 위해 dh 전달

