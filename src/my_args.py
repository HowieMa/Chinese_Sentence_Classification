import argparse


def build_args_parser():
    parser = argparse.ArgumentParser(description='TextCNN text classifier')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=2560, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stopping', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
    parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                        help='comma-separated filter sizes to use for convolution')
    parser.add_argument('-sen_len', type=int, default=32, help='max length of sentence')

    parser.add_argument('-static', type=bool, default=False, help='whether to use static pre-trained word vectors')
    parser.add_argument('-non-static', type=bool, default=False,
                        help='whether to fine-tune static pre-trained word vectors')
    parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
    parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word',
                        help='filename of pre-trained word vectors')
    parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')

    # device
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')

    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    args = parser.parse_args()
    return args
