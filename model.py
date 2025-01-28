import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_len)
        hidden: (h0, c0) se desejar inicializar manualmente
        """
        embeds = self.embed(x)  # (batch_size, seq_len, embed_dim)
        out, hidden = self.lstm(embeds, hidden)  # (batch_size, seq_len, hidden_dim)
        logits = self.fc(out)  # (batch_size, seq_len, vocab_size)
        return logits, hidden
