import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus, word2id, id2word = ptb.load_data('train')
vocab_size = len(word2id)
corpus_size = len(corpus)

model = RnnlmGen()
model.load_params("../ch06/Rnnlm_reproduced.pkl")

start_word = "you"
start_id = word2id[start_word]
skip_words = ["N", "<unk>", "$"]
skip_ids = [word2id[element] for element in skip_words]

word_ids = model.generate(start_id, skip_ids)
txt = " ".join([id2word[i] for i in word_ids])
txt = txt.replace('<eos>', '.\n')
print(txt)