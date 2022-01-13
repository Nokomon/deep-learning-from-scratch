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
        self.cache = None   # 순잔파 시의 계산 결과를 잠시 유지하기 위한 변수

    def forward(self, h, idx):
        """
        h: 맥락 벡터 (은닉층 도달하기 전 contexts의 W_in 행벡터의 평균)
        idx: 정답/오답 인덱스
        """
        target_W = self.embed.forward(idx)   # idx: 배열 for 미니배치 처리
        out = np.sum(target_W * h, axis=1)   # 내적
        self.cache = (h, target_W)
        return out

    