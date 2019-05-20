import torch
from torch import nn
from torch.nn import functional as F


class RCNN(nn.Module):
    def __init__(self, args, hidden_size_linear=64, hidden_layers=2):
        super(RCNN, self).__init__()

        self.args = args

        class_num = args.class_num              # 3, 0 for unk, 1 for negative, 2 for postive
        vocabulary_size = args.vocabulary_size  # total number of vocab (2593)
        embedding_dimension = args.embedding_dim     # 128
        hidden_size = args.hidden_size               # 64

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

        # Bi-directional LSTM for Model_RCNN
        self.lstm = nn.LSTM(input_size=embedding_dimension,
                            hidden_size=hidden_size,
                            num_layers=hidden_layers,
                            dropout=args.dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(args.dropout)

        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(embedding_dimension + 2 * hidden_size, hidden_size_linear)

        # Tanh non-linearity
        self.tanh = nn.Tanh()

        # Fully-Connected Layer
        self.fc = nn.Linear(hidden_size_linear, class_num)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):

        """
        :param x: LongTensor    Batch_size * Sentence_length
       :return:
        """
        if self.embedding2:
            x = torch.cat([self.embedding(x), self.embedding2(x)], dim=2)  # bz * sen_len * (embed_dim * 2)
            x = x.permute(1, 0, 2)  # Sentence_length(32) * Batch_size * (embed_dim * 2)
        else:
            x = self.embedding(x)  # Batch_size * Sentence_length(32) * embed_dim(128)
            x = x.permute(1, 0, 2)  # Sentence_length(32) * Batch_size embed_dim(128)

        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out       Sentence_length * Batch_size * (hidden_layers * 2)

        input_features = torch.cat([lstm_out, x], dim=2).permute(1, 0, 2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)

        linear_output = self.tanh(self.W(input_features))
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)

        linear_output = linear_output.permute(0, 2, 1)  # Reshaping fot max_pool

        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)

        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        return self.softmax(final_out)
