import argparse
import asyncio
import time

import numpy as np
import torch
import torch.nn as nn
import websockets
from model_parts import Config
from torch.nn import functional as F
from transformers import GPT2Tokenizer  # Import the tokenizer


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    dims = tensor.shape
    num_dims = len(dims)
    header = num_dims.to_bytes(1, "big")
    for dim in dims:
        header += dim.to_bytes(4, "big")
    return header + tensor.numpy().tobytes()


def bytes_to_tensor(data: bytes) -> torch.Tensor:
    num_dims = int.from_bytes(data[0:1], "big")
    dims = []
    offset = 1
    for _ in range(num_dims):
        dims.append(int.from_bytes(data[offset : offset + 4], "big"))
        offset += 4
    shape = tuple(dims)
    tensor_data = np.frombuffer(data[offset:], dtype=np.float32)
    return torch.from_numpy(tensor_data).reshape(shape)


class OrchestratorModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward_initial(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.size()
        positions = torch.arange(0, t, device=x.device).unsqueeze(0)
        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(positions)
        return tok_emb + pos_emb

    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


async def generate_tokens(worker_ports, num_tokens_to_generate, config):
    model = OrchestratorModel(config)
    model.eval()

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    worker_uris = [f"ws://localhost:{port}" for port in worker_ports]
    connections = []

    try:
        for uri in worker_uris:
            connection = await websockets.connect(uri)
            connections.append(connection)

        print(f"Successfully connected to {len(connections)} workers.")

        # Use the specific context provided by the user
        initial_context = [15496, 11, 314, 1101, 257, 3303, 2746, 11]
        input_tokens = torch.tensor([initial_context], dtype=torch.long)
        generated_tokens = input_tokens

        start_time = time.monotonic()

        for i in range(num_tokens_to_generate):
            print(f"\n--- Iteration {i + 1}/{num_tokens_to_generate} ---")

            current_context = generated_tokens[:, -config.block_size :]

            # Decode and print the current context as text
            current_text = tokenizer.decode(current_context[0].tolist())
            print(f'Model Input (context): "{current_text}"')

            with torch.no_grad():
                x = model.forward_initial(current_context)

                for j, ws in enumerate(connections):
                    print(
                        f"Sending tensor to worker {j + 1} on port {worker_ports[j]}..."
                    )
                    await ws.send(tensor_to_bytes(x))
                    response_bytes = await ws.recv()
                    x = bytes_to_tensor(response_bytes)
                    print(f"Received processed tensor from worker {j + 1}.")

                logits = model.forward_final(x)

            logits_for_next_token = logits[:, -1, :]
            probabilities = F.softmax(logits_for_next_token, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Decode and print the predicted next token as text
            predicted_text = tokenizer.decode(next_token[0].tolist())
            print(f'Predicted next token: {next_token.item()} -> "{predicted_text}"')

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        end_time = time.monotonic()
        total_time = end_time - start_time

        # Decode the final generated sequence
        final_text = tokenizer.decode(generated_tokens[0].tolist())
        print(f"\nFinal generated text: \n---\n{final_text}\n---")
        print(f"\nTotal generation time: {total_time:.4f} seconds.")

    except Exception as e:
        print(f"An error occurred during orchestration: {e}")
    finally:
        for ws in connections:
            await ws.close()
        print("All worker connections have been closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for Distributed nanoGPT")
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=[8765, 8766, 8767, 8768, 8769, 8770],
        help="Ports of the worker servers",
    )
    parser.add_argument(
        "--num_tokens", type=int, default=10, help="Number of tokens to generate"
    )
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension")
    parser.add_argument(
        "--n_head", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--block_size", type=int, default=1024, help="Context length")
    args = parser.parse_args()

    config = Config(
        vocab_size=args.vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        block_size=args.block_size,
    )

    asyncio.run(
        generate_tokens(
            worker_ports=args.ports,
            num_tokens_to_generate=args.num_tokens,
            config=config,
        )
    )
