import torch
import torch.nn as nn
from torch.nn import functional as F
from config import BLOCK_SIZE, N_EMBED, DROPOUT, HEAD_SIZE, N_HEAD 

class Head(nn.Module):
    """ One masked self-attention head. """

    def __init__(self):
        super().__init__()
        self.key = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        # Causal mask buffer
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores (q @ k^T / sqrt(d_k))
        wei = q @ k.transpose(-2, -1) * (HEAD_SIZE ** -0.5)
        
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple attention heads in parallel followed by projection. """

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Concatenate outputs from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out