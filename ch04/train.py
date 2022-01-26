import sys
sys.path.append('..')

import numpy as np
from common import config
config.GPU = True

import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram
from common.util import *
from dataset import ptb

# hyperparameters
window_size = 5
hidden_size = 100
batch_size = 100

max_epoch = 10
# 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# 모델, 옵티마이저, 트레이너 설정
# CBOW
# model = CBOW(vocab_size, hidden_size, window_size, corpus)
model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 학습
trainer.fit(contexts, target, max_epoch, batch_size)

word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'skipgram_params_W_in.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
    
# loss plot 그리기 (파이썬 스크립트에서의 작동 검토)
trainer.plot()