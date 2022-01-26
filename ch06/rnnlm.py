import sys
sys.path.append('..')

from common.time_layers import *
import pickle

class Rnnlm:
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # Initialize weights
        # TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
        W_embed = (np.random.randn(V, D) / np.sqrt(V)).astype('f')
        Wx = ((np.random.randn(D, 4*H)) / np.sqrt(D)).astype('f')
        Wh = ((np.random.randn(H, 4*H)) / np.sqrt(H)).astype('f')
        b_lstm = np.zeros(4*H).astype('f')
        W_affine = ((np.random.randn(H, V)) / np.sqrt(H)).astype('f')
        b_affine