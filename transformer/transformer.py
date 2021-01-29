import torch
import torch.nn as nn


class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, d_model, label_shape):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(d_model, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, label_shape)
        self.relu = nn.ReLU()


    def forward(self, x, is_test=False):
        x = self.encoder(x) # attention value
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        if is_test: return x # (64, 200, 200)
        x = torch.sum(x, dim=1) # sum of 200 frames prediction
        return x
