import mlx.core as mx
import os
from tokenizers import Tokenizer
import os
from tokenizers import Tokenizer
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from bert.model import Bert, load_model
from dataloader import MLXDataloader
from dataset import MLXDataset
from mlx.optimizers import AdamW
from bert.convert import convert



def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling using NumPy."""
    def mask_tokens_helper(inputs, tokenizer, mlm_probability=0.15):

        labels = mx.array(inputs['input_ids'])
        mask_token_id = tokenizer.token_to_id(tokenizer.mask_token)
        vocab_size = tokenizer.get_vocab_size()

        probability_matrix = mx.random.uniform(shape=labels.shape) < mlm_probability

        masked_indices = probability_matrix

        replace_mask = mx.random.uniform(shape=labels.shape) < 0.8
        inputs['input_ids'] = mx.where(replace_mask & masked_indices, mx.array([mask_token_id] * labels.size).reshape(labels.shape), inputs['input_ids'])

        random_mask = mx.random.uniform(shape=labels.shape) < 0.1
        random_indices = random_mask & masked_indices & ~replace_mask
        random_words = mx.random.randint(low=0, high=vocab_size, shape=labels.shape)
        inputs['input_ids'] = mx.where(random_indices, random_words, inputs['input_ids'])

        return inputs, labels
    
    if isinstance(inputs, list):
        return [mask_tokens_helper(inp, tokenizer, mlm_probability) for inp in inputs]
    else:
        return mask_tokens_helper(inputs, tokenizer, mlm_probability)

def load_tokenizer():
    tokenizer = Tokenizer.from_file("eo_tokenizer.json") # huggingface tokenizers
    return tokenizer

if __name__ == "__main__":
    # Configuration
    model_name = 'bert-base-uncased'
    max_length = 128
    mlm_probability = 0.15
    batch_size = 16
    epochs = 3
    tokenizer = load_tokenizer()
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    # convert("bert-base-uncased", "./mlx/weights/bert-base-uncased.npz")
    model, _ = load_model("bert-base-uncased", "./mlx/weights/bert-base-uncased.npz")
    # print(type(model))
    model.train()

    with open("./eo.txt/eo_test.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
        texts = [text.strip() for text in texts if text.strip()]

    dataset = MLXDataset(texts, tokenizer)
    dataloader = MLXDataloader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(learning_rate=5e-5)
    
    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch
            out = mask_tokens(inputs, tokenizer, mlm_probability)
            print(out)
            # print("YIPPIE ", inputs, labels)
        break