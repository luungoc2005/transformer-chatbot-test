from os import path

VOCAB_PATH = './data/reddit/vocab/'

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_default_tokenizer():
    from tokenizers import SentencePieceBPETokenizer

    tokenizer = SentencePieceBPETokenizer(
        path.join(VOCAB_PATH, 'en-vocab.json'),
        path.join(VOCAB_PATH, 'en-merges.txt'),
        unk_token='[UNK]'
    )

    return tokenizer

def get_batch(source, i, bptt=64):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target