import sys
sys.path.append('..')

from common.time_layers import *
from common.base_model import BaseModel
import pickle

class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # Initialize weights
        # TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
        W_embed = (np.random.randn(V, D) / np.sqrt(V)).astype('f')
        Wx = ((np.random.randn(D, 4*H)) / np.sqrt(D)).astype('f')
        Wh = ((np.random.randn(H, 4*H)) / np.sqrt(H)).astype('f')
        b_lstm = np.zeros(4*H).astype('f')
        W_affine = ((np.random.randn(H, V)) / np.sqrt(H)).astype('f')
        b_affine = np.zeros(V).astype('f')

        # Create layers
        self.layers = [
            TimeEmbedding(W_embed),
            TimeLSTM(Wx, Wh, b_lstm),
            TimeAffine(W_affine, b_affine)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        # Gather all params and grads into one
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        prediction = self.predict(xs)
        loss = self.loss_layer.forward(prediction, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()

    # def load_save_params(self, mode, file_name='Rnnlm.pkl'):
    #     if mode == "save":
    #         with open(file_name, 'wb') as f:
    #             pickle.dump(self.params, f, -1)
    #
    #     elif mode == "load":
    #         with open(file_name, 'rb') as f:
    #             self.params = pickle.load(f)

