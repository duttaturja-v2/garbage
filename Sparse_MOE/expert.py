import torch
import torch.nn as nn
from torch.nn import functional as F
from config import N_EMBED, DROPOUT, TOP_K, NUM_EXPERTS

class Expert(nn.Module):
    """ A simple MLP that acts as an Expert in the MoE layer. """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    """ Implements the Noisy Top-K Gating mechanism to select experts. """
    def __init__(self):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        self.topkroute_linear = nn.Linear(N_EMBED, NUM_EXPERTS)
        self.noise_linear = nn.Linear(N_EMBED, NUM_EXPERTS)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        
        # Add scaled unit Gaussian noise
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        # Select the top-k experts
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        
        # Create sparse tensor for softmax by setting unselected logits to -inf
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        
        router_output = F.softmax(sparse_logits, dim=-1)
        
        # router_output shape: [B, T, NUM_EXPERTS] (sparse with TOP_K non-zero values)
        # indices shape: [B, T, TOP_K] (indices of the selected experts)
        return router_output, indices