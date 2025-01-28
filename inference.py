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
    word2idx_local = {word: idx for idx, word in enumerate(vocab)}
    idx2word_local = {idx: word for word, idx in word2idx_local.items()}
    return word2idx_local, idx2word_local

def load_model(model_filepath, vocab_size_local, embed_dim_local, hidden_dim_local, num_layers_local):
    """
    Load the trained model from a file.
    """
    model_local = LSTMModel(vocab_size_local, embed_dim_local, hidden_dim_local, num_layers=num_layers_local)
    model_local.load_state_dict(torch.load(model_filepath))
    model_local.eval()
    return model_local

def generate_text(model, start_text, word2idx, idx2word, predict_len=10):
    """
    Generate text using the trained model.
    """
    tokens = start_text.lower().split()
    input_ids = [word2idx.get(w, 0) for w in tokens]  # Use idx=0 if word not found
    input_tensor = torch.LongTensor([input_ids])
    hidden = None

    generated = tokens[:]  # Copy tokens
    with torch.no_grad():
        for _ in range(predict_len):
            logits, hidden = model(input_tensor, hidden)
            # Take the logits of the last position
            last_logits = logits[0, -1, :]  # shape: (vocab_size,)
            probs = torch.softmax(last_logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            next_word = idx2word[next_idx]
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
    START_TEXT = "buy stocks"

    # Load vocabulary and model
    word2idx, idx2word = load_vocab(VOCAB_PATH)
    vocab_size = len(word2idx)
    lstm_model = load_model(MODEL_PATH, vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)

    # Generate text
    generated_text = generate_text(lstm_model, START_TEXT, word2idx, idx2word, predict_len=10)
    print("Generated text:", generated_text)
