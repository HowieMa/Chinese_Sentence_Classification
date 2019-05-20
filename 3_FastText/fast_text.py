import torch
from torch import nn


class FastText(nn.Module):
    def __init__(self, args):
        super(FastText, self).__init__()

        class_num = args.class_num    # 3, 0 for unk, 1 for negative, 2 for postive
        vocabulary_size = args.vocabulary_size  # total number of vocab (2593)
        embedding_dimension = args.embedding_dim     # 128
        hidden_size = args.hidden_size
        # Embedding Layer
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        # **************** different kinds of initialization for embedding ****************
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)

        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            embedding_dimension *= 2
        else:
            self.embedding2 = None

        # ********************* model layer *********************
        # Hidden Layer
        self.fc1 = nn.Linear(embedding_dimension, hidden_size)
        # Output Layer
        self.fc2 = nn.Linear(hidden_size, class_num)
        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        :param x: LongTensor    Batch_size * Sentence_length
        :return:
        """
        if self.embedding2:
            x = torch.cat([self.embedding(x), self.embedding2(x)], dim=2)  # bz * sen_len * (embed_dim * 2)
        else:
            x = self.embedding(x)       # Batch_size * Sentence_length(32) * embed_dim(128)

        x = x.mean(dim=1)               # Batch_size * Embed_dim(128)
        x = self.fc1(x)                 # Batch_size * Hidden(128)
        x = self.fc2(x)
        return self.softmax(x)
