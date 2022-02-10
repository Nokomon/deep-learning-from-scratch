import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

# read dataset
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char2id, id2char = sequence.get_vocab()

reverse_input = True
if reverse_input:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# hyperparameters
vocab_size = len(char2id)
vector_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5

# model, optimizer, trainer
# model = Seq2seq(vocab_size, vector_size, hidden_size)
model = PeekySeq2seq(vocab_size, vector_size, hidden_size)
# test
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]   # (1, n) shape의 array로 받음
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id2char, verbose)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print(f'Evaluation Accuracy: {acc * 100:.3f}')