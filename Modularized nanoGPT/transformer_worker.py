import argparse
import asyncio

import numpy as np
import torch
import websockets
from model_parts import Block, Config

transformer_block = None


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


async def handler(websocket):
    global transformer_block
    print(f"Connection established with {websocket.remote_address}")
    try:
        async for message in websocket:
            received_tensor = bytes_to_tensor(message)
            print(f"Received tensor of shape: {received_tensor.shape}")

            with torch.no_grad():
                output_tensor = transformer_block(received_tensor)

            print(f"Processed tensor, output shape: {output_tensor.shape}")

            response_bytes = tensor_to_bytes(output_tensor)
            await websocket.send(response_bytes)
            print("Sent processed tensor back to orchestrator.")

    except websockets.exceptions.ConnectionClosedError:
        print(f"Connection closed with {websocket.remote_address}")
    except Exception as e:
        print(f"An error occurred: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Transformer Worker Server for nanoGPT"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port to run the WebSocket server on"
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

    global transformer_block
    transformer_block = Block(config)
    transformer_block.eval()

    async with websockets.serve(handler, "localhost", args.port):
        print(f"Transformer worker running on ws://localhost:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server is shutting down.")
