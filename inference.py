"""
inference.py

This script loads a trained LSTM model for text generation and allows users to generate text
based on a given start phrase.
"""

import os
import torch
from model import LSTMModel

def load_vocab(vocab_filepath):
    """
    Load vocabulary from file.
    """
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
        vocab = f.read().splitlines()
    local_word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    local_idx_to_word = {idx: word for word, idx in local_word_to_idx.items()}
    return local_word_to_idx, local_idx_to_word

def load_model(
    model_filepath, vocab_size, embed_dim, hidden_dim, num_layers
):
    """
    Load the trained model from a file.
    """
    local_model = LSTMModel(
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers=num_layers
    )
    local_model.load_state_dict(torch.load(model_filepath))
    local_model.eval()
    return local_model

def generate_text(
    lstm_model, start_phrase, word_to_idx_map, idx_to_word_map, predict_len=10
):
    """
    Generate text using the trained model.
    """
    tokens = start_phrase.lower().split()
    input_ids = [word_to_idx_map.get(w, 0) for w in tokens]  # Use idx=0 if word not found
    input_tensor = torch.LongTensor([input_ids])
    hidden = None

    generated = tokens[:]  # Copy tokens
    with torch.no_grad():
        for _ in range(predict_len):
            logits, hidden = lstm_model(input_tensor, hidden)
            # Take the logits of the last position
            last_logits = logits[0, -1, :]  # shape: (vocab_size,)
            probs = torch.softmax(last_logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            next_word = idx_to_word_map[next_idx]
            generated.append(next_word)

            # Update input_tensor to include the new token
            input_ids.append(next_idx)
            input_tensor = torch.LongTensor([input_ids])

    return " ".join(generated)

if __name__ == "__main__":
    # Constants
    MODEL_PATH = os.path.join("data", "trained_model.pth")
    VOCAB_PATH = os.path.join("data", "vocab.txt")
    EMBED_DIM = 32
    HIDDEN_DIM = 64
    NUM_LAYERS = 1
    START_PHRASE = "buy stocks"

    # Load vocabulary and model
    GLOBAL_WORD_TO_IDX, GLOBAL_IDX_TO_WORD = load_vocab(VOCAB_PATH)
    VOCAB_SIZE = len(GLOBAL_WORD_TO_IDX)
    LSTM_MODEL = load_model(MODEL_PATH, VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)

    # Generate text
    GENERATED_TEXT = generate_text(
        LSTM_MODEL, START_PHRASE, GLOBAL_WORD_TO_IDX, GLOBAL_IDX_TO_WORD, predict_len=10
    )
    print("Generated text:", GENERATED_TEXT)
