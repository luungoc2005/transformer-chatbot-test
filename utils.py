from os import path

VOCAB_PATH = './data/reddit/vocab/'

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_default_tokenizer():
    from tokenizers import SentencePieceBPETokenizer
    from data_reddit.generate_vocab import special_tokens

    tokenizer = SentencePieceBPETokenizer(
        path.join(VOCAB_PATH, 'en-vocab.json'),
        path.join(VOCAB_PATH, 'en-merges.txt'),
        unk_token='[UNK]'
    )
    tokenizer.add_special_tokens(special_tokens)

    return tokenizer
