"""
train.py

This script trains an LSTM model for text generation using a given dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMModel


def load_data(file_path: str, seq_len: int = 5):
    """
    Reads a text file, creates a vocabulary, and generates (x, y) pairs for language modeling.

    Args:
        file_path (str): Path to the dataset file.
        seq_len (int): Number of tokens in each input sequence.

    Returns:
        tuple: Tensors for input (x), target (y), vocabulary, word-to-index, and index-to-word mappings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    tokens = text.split()
    vocab = sorted(set(tokens))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    data_ids = [word2idx[w] for w in tokens]

    xs, ys = [], []
    for i in range(len(data_ids) - seq_len):
        x_seq = data_ids[i:i + seq_len]
        y_seq = data_ids[i + 1:i + seq_len + 1]  # next token
        xs.append(x_seq)
        ys.append(y_seq)

    return (
        torch.LongTensor(xs),
        torch.LongTensor(ys),
        vocab,
        word2idx,
        idx2word,
    )


def train_language_model(
    data_file: str,
    seq_len: int = 5,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 1,
    lr: float = 0.01,
    epochs: int = 5,
    batch_size: int = 8,
    model_save_path: str = "trained_model.pth"
):
    """
    Train an LSTM language model.

    Args:
        data_file (str): Path to the dataset file.
        seq_len (int): Number of tokens in each input sequence.
        embed_dim (int): Embedding dimension.
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of LSTM layers.
        lr (float): Learning rate.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        model_save_path (str): Path to save the trained model.

    Returns:
        tuple: Trained model, vocabulary, word-to-index, and index-to-word mappings.
    """
    xs, ys, vocab, word2idx, idx2word = load_data(data_file, seq_len=seq_len)
    vocab_size = len(vocab)

    model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset_size = xs.size(0)
    num_batches = dataset_size // batch_size

    for epoch in range(epochs):
        total_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = xs[start:end]
            y_batch = ys[start:end]

            # Forward pass
            logits, _ = model(x_batch)
            loss = criterion(
                logits.view(-1, vocab_size),
                y_batch.view(-1)
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, vocab, word2idx, idx2word


def generate_text(
    model: LSTMModel,
    start_text: str,
    word2idx: dict,
    idx2word: dict,
    predict_len: int = 10
):
    """
    Generate text using a trained LSTM model.

    Args:
        model (LSTMModel): Trained LSTM model.
        start_text (str): Initial phrase to start text generation.
        word2idx (dict): Word-to-index mapping.
        idx2word (dict): Index-to-word mapping.
        predict_len (int): Number of tokens to generate.

    Returns:
        str: Generated text.
    """
    model.eval()
    tokens = start_text.lower().split()
    input_ids = [word2idx.get(w, 0) for w in tokens]  # Use idx=0 if word not found
    input_tensor = torch.LongTensor([input_ids])
    hidden = None

    generated = tokens[:]  # Copy tokens
    with torch.no_grad():
        for _ in range(predict_len):
            logits, hidden = model(input_tensor, hidden)
            last_logits = logits[0, -1, :]  # shape: (vocab_size,)
            probs = torch.softmax(last_logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            next_word = idx2word[next_idx]
            generated.append(next_word)

            input_ids.append(next_idx)
            input_tensor = torch.LongTensor([input_ids])

    return " ".join(generated)


if __name__ == "__main__":
    DATA_FILE = os.path.join("data", "frases_acoes.txt")
    MODEL_PATH = os.path.join("data", "trained_model.pth")

    # Train the model
    MODEL, VOCAB, W2I, I2W = train_language_model(
        data_file=DATA_FILE,
        seq_len=5,
        embed_dim=32,
        hidden_dim=64,
        num_layers=1,
        lr=0.01,
        epochs=5,
        batch_size=8,
        model_save_path=MODEL_PATH
    )

    # Generate text
    START_TEXT = "compra"
    GENERATED_TEXT = generate_text(MODEL, START_TEXT, W2I, I2W, predict_len=10)
    print("Generated text:", GENERATED_TEXT)
