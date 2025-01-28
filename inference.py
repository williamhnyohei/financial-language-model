import torch
from model import LSTMModel
import os

def load_vocab(vocab_file):
    """
    Load vocabulary from file.
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = f.read().splitlines()
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def load_model(model_path, vocab_size, embed_dim, hidden_dim, num_layers):
    """
    Load the trained model from a file.
    """
    model = LSTMModel(vocab_size, embed_dim, hidden_dim, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_text(model, start_text, word2idx, idx2word, predict_len=10):
    """
    Generate text using the trained model.
    """
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
    # Paths
    model_path = os.path.join("data", "trained_model.pth")
    vocab_file = os.path.join("data", "vocab.txt")

    # Load vocabulary and model
    word2idx, idx2word = load_vocab(vocab_file)
    vocab_size = len(word2idx)
    embed_dim = 32
    hidden_dim = 64
    num_layers = 1
    model = load_model(model_path, vocab_size, embed_dim, hidden_dim, num_layers)

    # Run inference
    start_text = "buy stocks"
    generated_text = generate_text(model, start_text, word2idx, idx2word, predict_len=10)
    print("Generated text:", generated_text)
