import torch
from torch import nn


class TextRNN(nn.Module):
    def __init__(self, args, hidden_layers=2):
        super(TextRNN, self).__init__()
        self.args = args

        class_num = args.class_num              # 3, 0 for unk, 1 for negative, 2 for postive
        vocabulary_size = args.vocabulary_size  # total number of vocab (2593)
        embedding_dimension = args.embedding_dim     # 128
        hidden_size = args.hidden_size                  # 32

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

        self.lstm = nn.LSTM(input_size=embedding_dimension,
                            hidden_size=hidden_size,
                            num_layers=hidden_layers,
                            dropout=args.dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)

        # Fully-Connected Layer
        self.fc = nn.Linear(hidden_size * hidden_layers * 2, class_num)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        """
        :param x: LongTensor    Batch_size * Sentence_length
       :return:
        """
        if self.embedding2:
            x = torch.cat([self.embedding(x), self.embedding2(x)], dim=2)  # bz * sen_len * (embed_dim * 2)
            x = x.permute(1, 0, 2)  # Sentence_length(32) * Batch_size * (embed_dim(128)*2)
        else:
            x = self.embedding(x)       # Batch_size * Sentence_length(32) * embed_dim(128)
            x = x.permute(1, 0, 2)      # Sentence_length(32) * Batch_size * embed_dim(128)

        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out       Sentence_length * Batch_size * (hidden_layers * 2 [bio-direct])
        # h_n           （num_layers * 2） * Batch_size * hidden_layers

        feature_map = self.dropout(h_n)  # （num_layers*2） * Batch_size * hidden_layers
        feature_map = torch.cat([feature_map[i, :, :] for i in range(feature_map.shape[0])], dim=1)
        # Batch_size * (hidden_size * hidden_layers * 2)
        final_out = self.fc(feature_map)    # Batch_size * class_num
        return self.softmax(final_out)
