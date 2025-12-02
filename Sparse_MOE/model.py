import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from config import N_EMBED, N_LAYER, BLOCK_SIZE, DEVICE
from block import Block


class SparseMoELanguageModel(nn.Module):
    """ The main Sparse Mixture of Experts Language Model (GPT-style). """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        
        # Stack of MoE Transformer Blocks
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        """Kaiming initialization for linear layers."""
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Generates new tokens based on the initial context (idx)."""
        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens for context
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            # Get the predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx