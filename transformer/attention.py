import numpy as np
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    ''' Generate Q, K, V '''
    
    def __init__(self,  d_m, d_k, d_v, dropout=0.1):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_m, d_k)
        self.w_k = nn.Linear(d_m, d_k)
        self.w_v = nn.Linear(d_m, d_v)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (d_m + d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (d_m + d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (d_m + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)

        self.fc = nn.Linear(d_v, d_m)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_m)



    def forward(self, q, k, v):

        d_k, d_v = self.d_k, self.d_v

        sz_b, len_q, d_m = q.size() # (64, 200, 512) 512 is d_m
        sz_b, len_k, d_m = k.size()
        sz_b, len_v, d_m = v.size()

        residual = q

        q = self.w_q(q) # (64, 200, 512) 512 is d_k
        k = self.w_k(k)
        v = self.w_v(v)

        attn = self.attention(q, k, v) # scaled dot-product attention

        attn = self.dropout(self.fc(attn)) # d_v to d_m
        attn = self.layer_norm(attn + residual) # residual connection

        return attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2)) # (64, 200, 200)
        attn = attn / self.temperature
        attn = self.softmax(attn) 
        attn = torch.bmm(attn, v) # (64, 200, 512) 512 is d_v
        return attn