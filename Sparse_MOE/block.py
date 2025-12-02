import torch
import torch.nn as nn
from config import N_EMBED, N_HEAD, NUM_EXPERTS, TOP_K, CAPACITY_FACTOR
from attention import MultiHeadAttention
from expert import Expert, NoisyTopkRouter


class SparseMoE(nn.Module):
    """ The Sparse Mixture of Experts layer. """
    def __init__(self):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter()
        self.experts = nn.ModuleList([Expert() for _ in range(NUM_EXPERTS)])
        self.top_k = TOP_K
        self.capacity_factor = CAPACITY_FACTOR
        self.num_experts = NUM_EXPERTS
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Get routing probabilities (gating_output) and selected expert indices
        gating_output, indices = self.router(x)
        
        # Flatten the input and gating outputs for easy indexing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Calculate expert capacity
        tokens_per_batch = batch_size * seq_len * self.top_k
        expert_capacity = int((tokens_per_batch / self.num_experts) * self.capacity_factor)
        
        flat_indices = indices.view(-1, self.top_k)
        updates = torch.zeros_like(flat_x)

        # Iterate over each expert
        for i, expert in enumerate(self.experts):
            # Identify which tokens are routed to expert 'i'
            expert_mask = (flat_indices == i)
            # Get the indices of the tokens (in the B*T space) routed to expert 'i'
            selected_indices = torch.nonzero(expert_mask.any(dim=-1)).squeeze(-1)
            
            # Apply Capacity Limit
            limited_indices = selected_indices[:expert_capacity]
            
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                
                # Get the routing score for the selected expert 'i'
                gating_scores = flat_gating_output[limited_indices, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                
                # Accumulate weighted output back to the original token positions
                updates.index_add_(0, limited_indices, weighted_output)

        # Reshape and return
        final_output = updates.view(batch_size, seq_len, -1)

        return final_output


class Block(nn.Module):
    """ MoE Transformer Block: Self Attention followed by Sparse MoE (Pre-Norm). """

    def __init__(self):
        super().__init__()
        
        self.sa = MultiHeadAttention()
        self.smoe = SparseMoE()
        
        # Layer Norms
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        # x = x + Attention(LayerNorm(x))
        x = x + self.sa(self.ln1(x))
        # x = x + MoE(LayerNorm(x))
        x = x + self.smoe(self.ln2(x))
        return x