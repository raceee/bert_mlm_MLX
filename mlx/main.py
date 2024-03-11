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



def numpy_mask_tokens_np(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling using NumPy."""
    labels = mx.array(inputs['input_ids'])  # Clone input_ids using NumPy

    # Manually fetch special token IDs, falling back to default if unavailable
    cls_token_id = tokenizer.token_to_id(tokenizer.cls_token) if hasattr(tokenizer, 'cls_token') else None
    sep_token_id = tokenizer.token_to_id(tokenizer.sep_token) if hasattr(tokenizer, 'sep_token') else None
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token) if hasattr(tokenizer, 'pad_token') else None

    # Initialize the probability matrix with mlm_probability
    probability_matrix = mx.full(labels.shape, mlm_probability)
    
    # Manually create a special tokens mask
    special_tokens_mask = mx.array(
        [[tok in [cls_token_id, sep_token_id, pad_token_id] for tok in val] for val in labels.tolist()],
        dtype=bool
    )
    
    # Apply the special tokens mask
    probability_matrix[special_tokens_mask] = 0.0
    if pad_token_id is not None:
        padding_mask = labels == pad_token_id
        probability_matrix[padding_mask] = 0.0
    
    # Generate masked indices
    masked_indices = mx.random.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Set labels for unmasked tokens to -100

    # Mask 80% of the time
    indices_replaced = mx.random.bernoulli(mx.full(labels.shape, 0.8)).bool() & masked_indices
    inputs['input_ids'][indices_replaced] = tokenizer.token_to_id(tokenizer.mask_token)
    
    # Replace 10% of the time with random word
    indices_random = mx.random.bernoulli(mx.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = mx.random.randint(tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
    inputs['input_ids'][indices_random] = random_words[indices_random]

    return inputs, labels

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
    model, _ = load_model("bert-base-uncased", "weights/bert-base-uncased.npz")
    model.train()

    with open("./eo.txt/eo_test.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
        texts = [text.strip() for text in texts if text.strip()]

    dataset = MLXDataset(texts, tokenizer, max_length)
    dataloader = MLXDataloader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(learning_rate=1e-5)