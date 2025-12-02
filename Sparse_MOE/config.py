import torch
import os

# Configuration
MODEL_PATH = 'sparse_moe_model.pt'
INPUT_FILE = 'input.txt' 

# Hyperparameters
BATCH_SIZE = 16
BLOCK_SIZE = 32
MAX_ITERS = 5000
EVAL_INTERVAL = 100
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 400
DROPOUT = 0.1

# Architecture Parameters
N_EMBED = 128
N_HEAD = 8
N_LAYER = 8
HEAD_SIZE = N_EMBED // N_HEAD

# Sparse MoE Parameters
NUM_EXPERTS = 8
TOP_K = 2
CAPACITY_FACTOR = 1.0

# Seed for reproducibility
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: Input file '{INPUT_FILE}' not found. Please create it with training text.")