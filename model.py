"""
model.py

This module contains the LSTMModel class, which defines a simple LSTM-based model for language modeling tasks.
"""

from torch import nn
import torch

class LSTMModel(nn.Module):
    """
    A simple LSTM-based model for language modeling tasks.

    Attributes:
        embed: Embedding layer for token embeddings.
        lstm: LSTM layer for sequence modeling.
        fc: Fully connected layer for output logits.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        """
        Initialize the LSTMModel.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of the embedding vectors.
            hidden_dim (int): Number of hidden units in the LSTM.
            num_layers (int, optional): Number of LSTM layers. Default is 1.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
