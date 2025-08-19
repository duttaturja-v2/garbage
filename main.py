import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class Config:
    def __init__(self, Embedding_Dimension:int, Number_of_Head:int, Vocabulary_Size:int, Context_Length:int) -> None:
        assert Embedding_Dimension % Number_of_Head == 0
        self.Embedding_Dimension = Embedding_Dimension
        self.Number_of_Head = Number_of_Head
        self.Vocabulary_Size = Vocabulary_Size
        self.Context_Length = Context_Length
        self.Head_Dimension = self.Embedding_Dimension // self.Number_of_Head

class Multi_Layer_Perceptron(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.Linear_1 = nn.Linear(config.Embedding_Dimension, config.Embedding_Dimension * 4)
        self.Linear_2 = nn.Linear(config.Embedding_Dimension * 4, config.Embedding_Dimension)
        self.GELU = nn.GELU()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.Linear_1(x)
        x = self.GELU(x)
        x = self.Linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.Head_Dimension = config.Head_Dimension
        self.Number_of_Head = config.Number_of_Head
        self.Embedding_Dimension = config.Embedding_Dimension

        self.Query_Weights = nn.Linear(config.Embedding_Dimension, config.Embedding_Dimension)
        self.Key_Weights = nn.Linear(config.Embedding_Dimension, config.Embedding_Dimension)
        self.Value_Weights = nn.Linear(config.Embedding_Dimension, config.Embedding_Dimension)
        self.out_proj = nn.Linear(config.Embedding_Dimension, config.Embedding_Dimension)
        
        mask = torch.triu(torch.ones(config.Context_Length, config.Context_Length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        queries = self.Query_Weights(x)
        keys = self.Key_Weights(x)
        values = self.Value_Weights(x)

        queries = queries.view(b, n, self.Number_of_Head, self.Head_Dimension).transpose(1, 2)
        keys = keys.view(b, n, self.Number_of_Head, self.Head_Dimension).transpose(1, 2)
        values = values.view(b, n, self.Number_of_Head, self.Head_Dimension).transpose(1, 2)

        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.Head_Dimension)
        scores = scores.masked_fill(self.mask[:n, :n], float("-inf"))
        attention = F.softmax(scores, dim=-1)

        context = attention @ values
        context = context.transpose(1, 2).contiguous().view(b, n, self.Embedding_Dimension)
        
        return self.out_proj(context)

class Transformer(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.mlp = Multi_Layer_Perceptron(config)
        self.ln_1 = nn.LayerNorm(config.Embedding_Dimension)
        self.ln_2 = nn.LayerNorm(config.Embedding_Dimension)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Model(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.Embeddings = nn.Embedding(config.Vocabulary_Size, config.Embedding_Dimension)
        self.Positional_Encoding = nn.Embedding(config.Context_Length, config.Embedding_Dimension)
        
        self.Transformer1 = Transformer(config)
        self.Transformer2 = Transformer(config)
        self.Transformer3 = Transformer(config)
        
        self.ln_final = nn.LayerNorm(config.Embedding_Dimension)
        self.Linear_final = nn.Linear(config.Embedding_Dimension, config.Vocabulary_Size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b, n = x.shape
        positions = torch.arange(0, n, device=x.device)

        token_embeddings = self.Embeddings(x)
        positional_embeddings = self.Positional_Encoding(positions)
        
        x = token_embeddings + positional_embeddings
        
        x = self.Transformer1(x)
        x = self.Transformer2(x)
        x = self.Transformer3(x)
        
        x = self.ln_final(x)
        logits = self.Linear_final(x)
        
        return logits



if __name__ == "__main__":
    config = Config(
        Vocabulary_Size=50,
        Embedding_Dimension=8,
        Number_of_Head=4,
        Context_Length=32
    )

    model = Model(config)
    model.eval() 

    input_tokens = torch.randint(0, config.Vocabulary_Size, (1, 5))
    
    generated_tokens = input_tokens
    num_tokens_to_generate = 10

    for i in range(num_tokens_to_generate):

        current_context = generated_tokens[:, -config.Context_Length:]
        print(f"Model Input (context): {current_context.tolist()}")
        
        with torch.no_grad():
            logits = model(current_context)
        
        logits_for_next_token = logits[:, -1, :] 
        
        probabilities = F.softmax(logits_for_next_token, dim=-1)
        
        next_token = torch.multinomial(probabilities, num_samples=1)
        print(f"Predicted next token: {next_token.item()}")
        
        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        print(f"Sequence for next step: {generated_tokens.tolist()}\n")