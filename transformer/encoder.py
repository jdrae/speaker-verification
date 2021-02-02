import torch
import torch.nn as nn
import numpy as np
from .attention import SelfAttention
from .ffn import PositionwiseFeedForward

class Encoder(nn.Module):
    def __init__(self, d_input, n_layers, d_k, d_v, d_m, d_ff, dropout=0.1):
        super().__init__()
        self.d_input = d_input
        self.n_layers = n_layers
        self.d_k = d_k
        self.d_v = d_v
        self.d_m = d_m
        self.d_ff = d_ff

        self.conv1 = nn.Conv1d(d_input, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)

        self.layer_norm_in = nn.LayerNorm(d_m)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_m, d_ff, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        # print(x.shape) # (64,13,200)
        x = self.conv1(x) # (64, 256, 200)
        x = self.conv2(x) # (64, 512, 200)
        x = x.transpose(1,2) # (64, 200, 512)

        output = self.dropout(self.layer_norm_in(x)) # (64, 200, 512)

        for enc_layer in self.layer_stack:
            output = enc_layer(output)

        return output


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_m, d_ff, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = SelfAttention(d_m, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_m, d_ff, dropout=dropout)

    def forward(self, enc_input):
        enc_slf_attn = self.slf_attn(enc_input)
        enc_output = self.pos_ffn(enc_slf_attn)

        return enc_slf_attn