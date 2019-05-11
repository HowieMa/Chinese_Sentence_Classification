import torch
from torch import nn
import numpy as np


class fastText(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(fastText, self).__init__()
        self.config = config

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        # Hidden Layer
        self.fc1 = nn.Linear(self.config.embed_size, self.config.hidden_size)

        # Output Layer
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.output_size)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1, 0, 2)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)
        return self.softmax(z)
