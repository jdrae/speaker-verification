import torch
import torch.nn as nn


class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, pooling, d_m, label_shape, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.fc1 = nn.Linear(d_m, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, label_shape)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, is_test=False):
        x = self.encoder(x) # attention value (bs, T, d_m)
        x = self.pooling(x) # (bs, 1, d_m)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x) #(bs, 1, fc2_feature)
        x = self.dropout(x)

        if is_test:
            return torch.squeeze(x) # (bs, fc2_feature)
        
        x = self.fc3(x)
        x = self.relu(x) # (bs, 1, label_num)
        return torch.squeeze(x) #(bs, label_num)
