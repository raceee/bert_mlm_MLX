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
        labels = mx.array(inputs['input_ids'])  # Clone input_ids using NumPy

        # Manually fetch special token IDs, falling back to default if unavailable
        cls_token_id = tokenizer.token_to_id(tokenizer.cls_token) if hasattr(tokenizer, 'cls_token') else None
        sep_token_id = tokenizer.token_to_id(tokenizer.sep_token) if hasattr(tokenizer, 'sep_token') else None
        pad_token_id = tokenizer.token_to_id(tokenizer.pad_token) if hasattr(tokenizer, 'pad_token') else None

        # Initialize the probability matrix with mlm_probability
        probability_matrix = mx.full(labels.shape, mlm_probability)
        
        # Manually create a special tokens mask
        special_tokens_mask = mx.array(
            [[tok in [cls_token_id, sep_token_id, pad_token_id] for tok in val] for val in labels.tolist()])

        # Apply the special tokens mask
        probability_matrix = mx.array([0.0 if m else val for val, m in zip(probability_matrix, special_tokens_mask[0])])
        
        if pad_token_id is not None:
            padding_mask = labels == pad_token_id
            probability_matrix = mx.array([0.0 if m else val for val, m in zip(probability_matrix, special_tokens_mask[0])])
        
        # Generate masked indices
        masked_indices = mx.random.bernoulli(probability_matrix)
        print("MASKED INDICES ", masked_indices.shape, probability_matrix.shape)
        labels = mx.array([-100 if not m else val for val, m in zip(labels[0], masked_indices[0])])

        # Mask 80% of the time
        indices_replaced = mx.random.bernoulli(mx.full(labels.shape, 0.8)).bool() & masked_indices
        inputs['input_ids'][indices_replaced] = tokenizer.token_to_id(tokenizer.mask_token)
        
        # Replace 10% of the time with random word
        indices_random = mx.random.bernoulli(mx.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = mx.random.randint(tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
        inputs['input_ids'][indices_random] = random_words[indices_random]

        return inputs, labels
    
    if isinstance(inputs, list):
        return [mask_tokens_helper(inp, tokenizer, mlm_probability) for inp in inputs]
    else:
        return mask_tokens_helper(inputs, tokenizer, mlm_probability)

# 1. Load the tokenizer
def load_tokenizer():
    tokenizer = Tokenizer.from_file("eo_tokenizer.json") # huggingface tokenizers
    return tokenizer

# 2. Convert model with the script in bert/convert.py


# 3. implemented the masked language model dataset
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
            inputs, labels = mask_tokens(inputs, tokenizer, mlm_probability)
            # print("YIPPIE ", inputs, labels)
        break