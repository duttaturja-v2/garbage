import torch
import os
from config import BLOCK_SIZE, BATCH_SIZE, DEVICE, INPUT_FILE, TORCH_SEED

torch.manual_seed(TORCH_SEED)

# Global variables for vocabulary and data
chars = []
vocab_size = 0
stoi = {}
itos = {}
encode = None
decode = None
train_data = None
val_data = None


def load_data_and_create_vocab():
    """Loads text data, creates vocabulary mappings, and splits data."""
    global chars, vocab_size, stoi, itos, encode, decode, train_data, val_data

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"The input file '{INPUT_FILE}' was not found. Please create it.")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Vocabulary size: {vocab_size} characters.")


def get_batch(split):
    """Generates a batch of inputs x and targets y."""
    data = train_data if split == 'train' else val_data
    if data is None:
        raise RuntimeError("Data not loaded. Call load_data_and_create_vocab() first.")
        
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y