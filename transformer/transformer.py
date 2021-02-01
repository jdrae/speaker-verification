import torch
import torch.nn as nn


class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, pooling, d_m, label_shape):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.fc1 = nn.Linear(d_m, 330)
        self.fc2 = nn.Linear(330, 220)
        self.fc3 = nn.Linear(220, label_shape)
        self.relu = nn.ReLU()


    def forward(self, x, is_test=False):
        x = self.encoder(x) # attention value (64, 200, 512)
        x = self.pooling(x) # (64, 1, 512)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x) #(64, 1, 220)
        if is_test:
            return torch.squeeze(x) # (64, 220)
        
        x = self.fc3(x)
        x = self.relu(x) # (64, 1, 196)
        return torch.squeeze(x) #(64, 196)
