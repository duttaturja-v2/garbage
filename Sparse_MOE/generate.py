import torch
import os
from config import DEVICE, MODEL_PATH
from data_utils import load_data_and_create_vocab, encode, decode, vocab_size
from model import SparseMoELanguageModel

def generate_text(start_string, max_tokens=200):
    """Loads the trained model and generates text based on a starting string."""
    try:
        load_data_and_create_vocab()
    except FileNotFoundError as e:
        print(f"Error: {e}. Cannot generate text without a vocabulary.")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run 'train.py' first.")
        return

    model = SparseMoELanguageModel(vocab_size)
    
    print(f"Loading trained model from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Error loading model state. Did the architecture parameters change? Error: {e}")
        return
        
    model.to(DEVICE)
    model.eval()

    if not all(c in encode for c in start_string):
        print("Warning: Starting string contains characters not in the training vocabulary.")
        # Simple error handling: replace unknown chars with a known one (e.g., first char)
        known_char = list(encode.keys())[0] if encode else 'a'
        start_string = ''.join([c if c in encode else known_char for c in start_string])
        
    context = torch.tensor(encode(start_string), dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    print("\n--- Generated Text ---")
    
    generated_indices = model.generate(context, max_tokens)
    generated_text = decode(generated_indices[0].tolist())
    
    print(generated_text)
    print("----------------------")


if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        prompt = "T"
        print(f"Generating text starting with: '{prompt}'")
        generate_text(prompt, max_tokens=300)
    else:
        print(f"Model file '{MODEL_PATH}' not found. Run 'python train.py' first, then run 'python generate.py'.")