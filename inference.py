"""
inference.py

This script loads a trained LSTM model for text generation and allows users to generate text
based on a given start phrase.
"""

import os
import torch
from model import LSTMModel


def load_vocab(vocab_filepath: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Load vocabulary from file.

    Args:
        vocab_filepath (str): Path to the vocabulary file.

    Returns:
        tuple: A tuple containing word-to-index and index-to-word mappings.
    """
    if not os.path.exists(vocab_filepath):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_filepath}")
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
        vocab = f.read().splitlines()
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def load_model(
    model_filepath: str, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int
) -> LSTMModel:
    """
    Load the trained model from a file.

    Args:
        model_filepath (str): Path to the model file.
        vocab_size (int): Vocabulary size.
        embed_dim (int): Embedding dimension.
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of LSTM layers.

    Returns:
        LSTMModel: The loaded LSTM model.
    """
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file not found: {model_filepath}")
    model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_layers=num_layers)
    model.load_state_dict(torch.load(model_filepath))
    model.eval()
    return model


def generate_text(
    lstm_model: LSTMModel,
    start_phrase: str,
    word_to_idx_map: dict[str, int],
    idx_to_word_map: dict[int, str],
    predict_len: int = 10,
) -> str:
    """
    Generate text using the trained model.

    Args:
        lstm_model (LSTMModel): The trained LSTM model.
        start_phrase (str): The initial phrase to start text generation.
        word_to_idx_map (dict): Word-to-index mapping.
        idx_to_word_map (dict): Index-to-word mapping.
        predict_len (int): Number of tokens to generate.

    Returns:
        str: Generated text.
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
    WORD_TO_IDX, IDX_TO_WORD = load_vocab(VOCAB_PATH)
    VOCAB_SIZE = len(WORD_TO_IDX)
    MODEL = load_model(MODEL_PATH, VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS)

    # Generate text
    GENERATED_TEXT = generate_text(
        MODEL, START_PHRASE, WORD_TO_IDX, IDX_TO_WORD, predict_len=10
    )
    print("Generated text:", GENERATED_TEXT)
