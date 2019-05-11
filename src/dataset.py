from torchtext.vocab import Vectors
from torchtext import data
import jieba


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def tokenizer_zh(x):
    """
    "The quick fox jumped over a lazy dog."   -> (tokenization)
    ["The", "quick", "fox", "jumped", "over", "a", "lazy", "dog", "."]
    """
    res = [w for w in jieba.cut(x)]
    return res


def build_stop_words_set(set_dir):
    stop_words = []
    with open(set_dir) as f:
        for l in f.readlines():
            stop_words.append(l.strip())
    return stop_words


def create_field(data_dir='../data/'):
    stop_words = build_stop_words_set(data_dir + 'stop_words.txt')
    text_field = data.Field(sequential=True, tokenize=tokenizer_zh, fix_length=1000, stop_words=stop_words)
    label_field = data.Field(sequential=False, use_vocab=False)
    return text_field, label_field


def get_dataset(text_field, label_field, data_dir='../data/'):
    train, valid, test = data.TabularDataset.splits(path=data_dir,
                                                    train='train.csv',
                                                    validation='valid.csv',
                                                    test='test.csv',
                                                    format='csv',
                                                    skip_header=False,
                                                    csv_reader_params={'delimiter': ','},
                                                    fields=[('text', text_field), ('label', label_field)]
                                                    )
    return train, valid, test


def load_dataset(text_field, label_field, data_dir, args, **kwargs):
    # ************************** get torch text dataset ***************************
    train_dataset, dev_dataset, test_dataset = get_dataset(text_field, label_field, data_dir=data_dir)

    # ************************** build vocabulary *********************************
    if args.static and args.pretrained_name and args.pretrained_path:
        # load pre-trained embedding vocab
        vectors = load_word_vectors(args.pretrained_name, args.pretrained_path)
        text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, dev_dataset)  # build vocab from train/val dataset only

    label_field.build_vocab(train_dataset, dev_dataset, min_freq=2)  # change from '0', '1' to 0,1

    vocab = text_field.vocab
    print(len(vocab))

    # **************************  build Iterator ***********************************
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset), args.batch_size),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter, dev_iter


if __name__ == "__main__":
    Text_field, Label_field = create_field()
    print(Text_field)               # torchtext.data.field.Field object
    print(Label_field)              # torchtext.data.field.Field object

    print('TEST Function: get_dataset ....')
    Train_dataset, Dev_dataset, Test_dataset = get_dataset(Text_field, Label_field)

    print(Train_dataset[0])         # torchtext.data.example.Example object
    print(Train_dataset[0].text)    # ['你', '快', '休息', '我爱你', '小度']
    print(Train_dataset[0].label)   # 1

