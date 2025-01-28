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

    def forward(self, x, hidden=None):
        """
        Forward pass of the LSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len).
            hidden (tuple, optional): Initial hidden state (h0, c0). Default is None.

        Returns:
            logits (Tensor): Output tensor of shape (batch_size, seq_len, vocab_size).
            hidden (tuple): Updated hidden state.
        """
        embeds = self.embed(x)  # (batch_size, seq_len, embed_dim)
        out, hidden = self.lstm(embeds, hidden)  # (batch_size, seq_len, hidden_dim)
        logits = self.fc(out)  # (batch_size, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state for the LSTM.

        Args:
            batch_size (int): The batch size.

        Returns:
            Tuple[Tensor, Tensor]: Initialized hidden state (h0, c0).
        """
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
        return h0, c0
