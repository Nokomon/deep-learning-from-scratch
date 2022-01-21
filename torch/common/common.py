import torch

class Embedding:
    def __init__(self, W):
        self.W = W
        self.params = [W]
        self.grads = torch.zeros_like(W)

    def forward(self, x):
