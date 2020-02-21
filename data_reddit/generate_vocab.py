from os import path, listdir
from tokenizers import SentencePieceBPETokenizer

DATA_PATH = './data/reddit/raw'
VOCAB_SIZE = 8000

special_tokens = [
    '[PAD]',
    '[UNK]',
    '[SEP]',
    '[P0]',
    '[P1]',
    '[DOC_SEP]'
]

tokenizer = SentencePieceBPETokenizer(unk_token='[UNK]')

texts = [
    path.join(DATA_PATH, item)
    for item in listdir(DATA_PATH)
    if item.endswith('.txt')
]

tokenizer.train(texts, 
    vocab_size=VOCAB_SIZE, 
    min_frequency=10,
    special_tokens=special_tokens
)

SAVE_PATH = path.join(DATA_PATH, 'vocab')
if not path.isdir(SAVE_PATH):
    import os
    os.makedirs(SAVE_PATH)

tokenizer.save(SAVE_PATH, 'en')
