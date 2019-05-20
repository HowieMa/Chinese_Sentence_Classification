import sys
import os
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)


from fast_text import *
from src.train import *
from src.my_args import *
from src.dataset import *


args = build_args_parser()
data_dir = args.dataset

print('Loading data Iterator ...')
text_field, label_field = create_field(args)

device = -1
train_iter, dev_iter, test_iter = load_dataset(text_field, label_field,
                                               args, device=-1, repeat=False, shuffle=True)


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

net = FastText(args)
if args.snapshot:
    print('\nLoading model from {}...\n'.format(args.snapshot))
    net.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    net = net.cuda()
try:
    train(train_iter, dev_iter, net, args)
except KeyboardInterrupt:
    print('Exiting from training early')

print('*'*30 + ' Testing ' + '*'*30)
save_prefix = os.path.join(args.save_dir, 'best')
save_path = '{}_model.pth'.format(save_prefix)
state_dict = torch.load(os.path.join(save_path))
net.load_state_dict(state_dict)
test_acc = test(test_iter, net, args)
