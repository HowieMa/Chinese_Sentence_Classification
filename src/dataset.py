from torchtext.vocab import Vectors
from torchtext import data
import jieba

# import sys
# import os
# currentUrl = os.path.dirname(__file__)
# parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
# sys.path.append(parentUrl)
# from my_args import *


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def tokenizer_zh(x):
    """
    "The quick fox jumped over a lazy dog."   -> (tokenization)
    ["The", "quick", "fox", "jumped", "over", "a", "lazy", "dog", "."]
    """
    res = [w for w in jieba.cut(x, cut_all=False)]
    return res


def build_stop_words_set(set_dir):
    stop_words = []
    with open(set_dir) as f:
        for l in f.readlines():
            stop_words.append(l.strip())
    return stop_words


def create_field(args):
    stop_words = build_stop_words_set(args.dataset + 'stop_words.txt')
    text_field = data.Field(sequential=True, tokenize=tokenizer_zh, fix_length=args.sen_len, stop_words=stop_words)
    label_field = data.Field(sequential=False)
    return text_field, label_field


def get_dataset(text_field, label_field, args):
    train, valid, test = data.TabularDataset.splits(path=args.dataset, format='tsv', skip_header=False,
                                                    train='train.tsv',
                                                    validation='valid.tsv',
                                                    test='test.tsv',
                                                    fields=[('text', text_field), ('label', label_field)]
                                                    )
    return train, valid, test


def load_dataset(text_field, label_field, args, **kwargs):
    # ************************** get torch text dataset ***************************
    train_dataset, dev_dataset, test_dataset = get_dataset(text_field, label_field, args)

    # ************************** build vocabulary *********************************
    if args.static and args.pretrained_name and args.pretrained_path:
        # load pre-trained embedding vocab
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)  # build vocab from train/val dataset only

    label_field.build_vocab(train_dataset, dev_dataset)  # change from '0', '1' to 0,1

    print('Num of class is ' + str(len(label_field.vocab)))
    print(label_field.vocab.stoi)

    # **************************  build Iterator ***********************************
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter, dev_iter, test_iter


# if __name__ == "__main__":
#
#     args = build_args_parser()
#
#     Text_field, Label_field = create_field(args)
#     print(Text_field)               # torchtext.data.field.Field object
#     print(Label_field)              # torchtext.data.field.Field object
#
#     print('TEST Function: get_dataset ....')
#     Train_dataset, Dev_dataset, Test_dataset = get_dataset(Text_field, Label_field, args)
#
#     max_len = -1
#     id = 0
#
#     for i in range(len(Train_dataset)):
#         if len(Train_dataset[i].text) > max_len:
#             max_len = len(Train_dataset[i].text)
#             id = i
#
#     print(id)
#     print('max length %d' % max_len)
#
#     print(Train_dataset[id].text)    # ['你', '快', '休息', '我爱你', '小度']
#     print(Train_dataset[id].label)   # 1
#
#     Train_iter, Dev_iter, Test_iter = load_dataset(Text_field, Label_field, args,
#                                                    device=-1, repeat=False, shuffle=True)
#     # Test_iter
#     batch = next(iter(Train_iter))
#     print(batch.text.shape)
#     print(batch.label.shape)
#
#     # vectors = load_word_vectors('sgns.zhihu.word', '../pretrained')
