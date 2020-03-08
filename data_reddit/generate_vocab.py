from os import path, listdir
from tokenizers import SentencePieceBPETokenizer

DATA_PATH = './data/reddit/raw'
SAVE_PATH = './data/reddit/vocab'
VOCAB_SIZE = 30522

special_tokens = [
    '[PAD]',
    '[UNK]',
    '[SEP]',
    '[P0]',
    '[P1]',
    '[DOC_SEP]'
]

if __name__ == "__main__":
    tokenizer = SentencePieceBPETokenizer(unk_token='[UNK]')

    texts = [
        path.join(DATA_PATH, item)
        for item in listdir(DATA_PATH)
        if item.endswith('.txt')
    ]

    tokenizer.train(texts, 
        vocab_size=VOCAB_SIZE, 
        min_frequency=2,
        special_tokens=special_tokens
    )

    if not path.isdir(SAVE_PATH):
        import os
        os.makedirs(SAVE_PATH)

    tokenizer.save(SAVE_PATH, 'en')
