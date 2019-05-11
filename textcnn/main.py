import torch
import torchtext.data as data


import model

from src import train
from src import dataset
from src import my_args


args = my_args.build_args_parser()


print('Loading data...')
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = dataset.load_dataset(text_field, label_field, args, device=-1, repeat=False, shuffle=True)

args.vocabulary_size = len(text_field.vocab)
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]

print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))

text_cnn = model.TextCNN(args)
if args.snapshot:
    print('\nLoading model from {}...\n'.format(args.snapshot))
    text_cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()
try:
    train.train(train_iter, dev_iter, text_cnn, args)
except KeyboardInterrupt:
    print('Exiting from training early')
