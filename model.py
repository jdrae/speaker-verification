import torch
import torch.nn.functional as F
from torch import nn, optim

class Logistic(nn.Module):
    def __init__(self, flat_shape, label_shape):
        super().__init__()
        self.fc1 = nn.Linear(flat_shape, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, label_shape) # (128,196)

    def forward(self, xb, is_test=False):
        xb = self.fc1(xb)
        xb = self.fc2(xb)
        xb = self.fc3(xb)
        xb = self.fc4(xb)
        if is_test: return xb
        pred = self.fc5(xb)
        return pred
