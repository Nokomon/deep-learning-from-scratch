import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 하이퍼파라미터
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5   # Truncated BPTT가 한 번에 펼치는 시간 크기
lr = 0.1
max_epoch = 100

# 학습 데이터 읽기(전체 중 1000개만)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)   # corpus는 0부터 매겨지기에 +1처리

xs = corpus[:-1]   # 입력: 맨 마지막값 제외. 맨 마지막값은 훈련에 안쓰이니까 (그 다음값이 없어서)
ts = corpus[1:]   # 정답 레이블: '정답'은 그 다음에 올 단어
data_size = len(xs)
print(f"말뭉치 크기: {corpus_size}, 어휘 수: {vocab_size}")

# 학습 시 사용하는 변수
max_iters = data_size // (batch_size * time_size)   # 19 (= 999 // (10 * 5))
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []   # 에폭마다 perplexity값 저장 용도

# 모델
model = SimpleRnnlm(vocab_size, hidden_size, wordvec_size)
optimizer = SGD(lr)

# 각 미니배치에서 샘플을 읽기 위해서 처리
jump = (corpus_size - 1) // batch_size   # 99: 어떤 값마다 시작위치를 잘라줄 것인가

# 미니배치 학습의 시작점들을 리스트로 담는다
offsets = [i*jump for i in range(batch_size)]   # 99, 198, ...

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 미니배치 획득
        batch_x = np.zeros((batch_size, time_size), dtype='i')
        batch_t = np.zeros((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # gradient 계산
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # 에폭마다 perplexity 평가
    perplexity = np.exp(total_loss / loss_count)   # e^L
    print(f"| Epoch {epoch+1} | Perplexity {perplexity}")
    ppl_list.append(float(perplexity))
    total_loss, loss_count = 0, 0

# 그래프 그리기
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
