from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import lzma

def open_eo():
    with lzma.open("./eo.txt.xz", 'rt', encoding='utf-8') as f:
        # read each line of the file
        for line in f:
            print(line, end='')
            break

def train_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(["./eo.txt/eo.txt"], trainer)
    tokenizer.save("eo_tokenizer.json")

def load_tokenizer():
    tokenizer = Tokenizer.from_file("eo_tokenizer.json") # huggingface tokenizers
    return tokenizer