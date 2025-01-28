import torch
import torch.nn as nn
import torch.optim as optim

import os
from model import LSTMModel

def load_data(file_path, seq_len=5):
    """
    Reads a text file, creates a vocabulary, and generates (x, y) pairs for language modeling.
    (x = seq_len tokens, y = tokens shifted by 1)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    tokens = text.split()
    vocab = sorted(list(set(tokens)))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    data_ids = [word2idx[w] for w in tokens]

    xs, ys = [], []
    for i in range(len(data_ids) - seq_len):
        x_seq = data_ids[i:i+seq_len]
        y_seq = data_ids[i+1:i+seq_len+1]  # next token
        xs.append(x_seq)
        ys.append(y_seq)

    xs = torch.LongTensor(xs)
    ys = torch.LongTensor(ys)

    return xs, ys, vocab, word2idx, idx2word

def train_language_model(
    data_file,
    seq_len=5,
    embed_dim=32,
    hidden_dim=64,
    num_layers=1,
    lr=0.01,
    epochs=5,
    batch_size=8
):
    # Load data
    xs, ys, vocab, word2idx, idx2word = load_data(data_file, seq_len=seq_len)
    vocab_size = len(vocab)

    # Create model
    model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Split into mini-batches
    dataset_size = xs.size(0)
    num_batches = dataset_size // batch_size

    for epoch in range(epochs):
        total_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = xs[start:end]
            y_batch = ys[start:end]

            # Forward
            logits, _ = model(x_batch)
            # logits: (batch_size, seq_len, vocab_size)
            # Flatten logits and y_batch to compare them
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
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    return model, vocab, word2idx, idx2word

def generate_text(
    model,
    start_text,
    word2idx,
    idx2word,
    predict_len=10
):
    """
    Generates text using the trained model.
    start_text: initial phrase (string), e.g., "buy"
    predict_len: number of additional tokens to generate
    """
    model.eval()
    tokens = start_text.lower().split()
    input_ids = [word2idx.get(w, 0) for w in tokens]  # use idx=0 if word not found
    input_tensor = torch.LongTensor([input_ids])
    hidden = None

    generated = tokens[:]  # copy tokens
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
    data_file = os.path.join("data", "frases_acoes.txt")
    model, vocab, w2i, i2w = train_language_model(
        data_file=data_file,
        seq_len=5,
        embed_dim=32,
        hidden_dim=64,
        num_layers=1,
        lr=0.01,
        epochs=5,
        batch_size=8
    )

    texto_inicial = "compra"
    texto_gerado = generate_text(model, texto_inicial, w2i, i2w, predict_len=10)
    print("Texto gerado:", texto_gerado)
