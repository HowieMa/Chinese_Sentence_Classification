import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        class_num = args.class_num    # 3, 0 for unk, 1 for negative, 2 for postive
        filter_num = args.filter_num  # 100
        filter_sizes = args.filter_sizes    # [3,4,5]
        vocabulary_size = args.vocabulary_size  # total number of vocab (2593)
        embedding_dimension = args.embedding_dim     # Default: 128 Pretrained: 300

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
        self.convs = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(in_channels=embedding_dimension,
                                        out_channels=filter_num,
                                        kernel_size=size),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=args.sen_len - size + 1))
                for size in filter_sizes
            ])

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        """
        :param x: LongTensor    Batch_size * Sentence_length
        :return:
        """
        if self.embedding2:
            x = torch.cat([self.embedding(x), self.embedding2(x)], dim=2)  # bz * sen_len * (embed_dim * 2)
            x = x.permute(0, 2, 1)  # Batch_size * (Embed_dim*2) * Sentence_length
        else:
            x = self.embedding(x)       # Batch_size * Sentence_length(32) * embed_dim(128)
            x = x.permute(0, 2, 1)      # Batch_size * Embed_dim(128) * Sentence_length(32)

        x = [conv(x).squeeze(2) for conv in self.convs]  # Batch_size * 100
        x = torch.cat(x, dim=1)  # Batch_size * 300(100+100+100)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
