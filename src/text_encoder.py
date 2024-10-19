from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.hidden_size = hidden_size

    def forward(self, captions):
        embedded = self.embedding(captions)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden[-1]
