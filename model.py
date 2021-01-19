import torch
import torch.nn.functional as F
from torch import nn, optim

class Logistic(nn.Module):
    def __init__(self, flat_shape, label_shape):
        super().__init__()
        self.lin1 = nn.Linear(flat_shape, 1500)
        self.lin2 = nn.Linear(1500, 1000)
        self.lin3 = nn.Linear(1000, 500)
        self.lin4 = nn.Linear(500, 250)
        self.lin5 = nn.Linear(250, label_shape)

    def forward(self, xb):
        xb = self.lin1(xb)
        xb = self.lin2(xb)
        xb = self.lin3(xb)
        xb = self.lin4(xb)
        pred = self.lin5(xb)
        return pred
        # return xb @ self.weights + self.bias
