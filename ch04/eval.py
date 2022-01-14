import sys
sys.path.append('..')

import numpy as np
from common import config
config.GPU = True

import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import *
from dataset import ptb

pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'rb') as f:
    params = pickle.load(f)   # keys: word_vecs, word_to_id, id_to_word
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)