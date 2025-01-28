"""
train.py

This script trains an LSTM model for text generation using a given dataset.
"""

import os
import torch
from torch import nn, optim
from model import LSTMModel


def load_data(file_path: str, seq_len: int = 5):
    """
    Reads a text file, creates a vocabulary, and generates (x, y) pairs for language modeling.
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
        xs.append(data_ids[i:i + seq_len])
        ys.append(data_ids[i + 1:i + seq_len + 1])

    return torch.LongTensor(xs), torch.LongTensor(ys), vocab, word2idx, idx2word


def train_batch(config: dict, x_batch, y_batch):
    """
    Train a single batch using the given configuration.
    """
    logits, _ = config["model"](x_batch)
    loss = config["criterion"](
        logits.view(-1, config["vocab_size"]),
        y_batch.view(-1)
    )
    config["optimizer"].zero_grad()
    loss.backward()
    config["optimizer"].step()
    return loss.item()


def train_epoch(config: dict, xs, ys):
    """
    Train the model for one epoch.
    """
    dataset_size = xs.size(0)
    num_batches = dataset_size // config["batch_size"]
    total_loss = 0.0

    for b in range(num_batches):
        start = b * config["batch_size"]
        end = start + config["batch_size"]
        x_batch, y_batch = xs[start:end], ys[start:end]
        total_loss += train_batch(config, x_batch, y_batch)

    return total_loss / num_batches


def train_language_model(config: dict):
    """
    Train an LSTM language model using the given configuration.
    """
    xs, ys, vocab, word2idx, idx2word = load_data(config["data_file"], seq_len=config["seq_len"])
    vocab_size = len(vocab)

    model = LSTMModel(vocab_size, config["embed_dim"], config["hidden_dim"], num_layers=config["num_layers"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    config.update({
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "vocab_size": vocab_size
    })

    for epoch in range(config["epochs"]):
        avg_loss = train_epoch(config, xs, ys)
        print(f"Epoch [{epoch + 1}/{config['epochs']}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Model saved to {config['model_save_path']}")
    return model, vocab, word2idx, idx2word


if __name__ == "__main__":
    CONFIG = {
        "data_file": os.path.join("data", "frases_acoes.txt"),
        "seq_len": 5,
        "embed_dim": 32,
        "hidden_dim": 64,
        "num_layers": 1,
        "lr": 0.01,
        "epochs": 5,
        "batch_size": 8,
        "model_save_path": os.path.join("data", "trained_model.pth"),
    }

    MODEL, VOCAB, W2I, I2W = train_language_model(CONFIG)
