# import mlx.core as mx
import os
from tokenizers import Tokenizer
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_tokenizer():
    tokenizer = Tokenizer.from_file("eo_tokenizer.json") # huggingface tokenizers
    return tokenizer

def tokenize_text(tokenizer, texts:list):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="np")

class MLM_Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def custom_encode_plus(self, text, padding='max_length', return_tensors=None):
        # Tokenize input text
        encoded = tokenizer.encode(text)
        
        # Truncate to max_length if specified
        tokens = encoded.ids[:self.max_length]
        
        # Create attention mask
        attention_mask = [1] * len(tokens)
        
        # Padding
        if padding == 'max_length':
            padding_length = self.max_length - len(tokens)
            tokens.extend([self.tokenizer.token_to_id("[PAD]")] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        # Prepare output
        output = {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
        
        # Convert to tensors if required
        if return_tensors == 'pt':
            output['input_ids'] = torch.tensor([output['input_ids']])
            output['attention_mask'] = torch.tensor([output['attention_mask']])
        
        return output


    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.custom_encode_plus(text,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        return inputs

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling."""
    labels = inputs['input_ids'].clone()
    
    # Attempt to manually fetch special token IDs, falling back to default if unavailable
    cls_token_id = tokenizer.token_to_id(tokenizer.cls_token) if hasattr(tokenizer, 'cls_token') else None
    sep_token_id = tokenizer.token_to_id(tokenizer.sep_token) if hasattr(tokenizer, 'sep_token') else None
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token) if hasattr(tokenizer, 'pad_token') else None
    
    # Initialize the probability matrix with mlm_probability
    probability_matrix = torch.full(labels.shape, mlm_probability)
    print("PROBABILTY MATRIX", probability_matrix.shape, probability_matrix)
    
    # Manually create a special tokens mask
    special_tokens_mask = torch.tensor(
        [[tok in [cls_token_id, sep_token_id, pad_token_id] for tok in val] for val in labels.tolist()], 
        dtype=torch.bool
    )
    
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    if pad_token_id is not None:
        padding_mask = labels.eq(pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    # Generate masked indices
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens
    
    # Mask 80% of the time
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs['input_ids'][indices_replaced] = tokenizer.token_to_id(tokenizer.mask_token)
    
    # Replace 10% of the time with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
    inputs['input_ids'][indices_random] = random_words[indices_random]

    return inputs, labels

import numpy as np

def numpy_mask_tokens_np(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling using NumPy."""
    labels = np.array(inputs['input_ids'])  # Clone input_ids using NumPy

    # Manually fetch special token IDs, falling back to default if unavailable
    cls_token_id = tokenizer.token_to_id(tokenizer.cls_token) if hasattr(tokenizer, 'cls_token') else None
    sep_token_id = tokenizer.token_to_id(tokenizer.sep_token) if hasattr(tokenizer, 'sep_token') else None
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token) if hasattr(tokenizer, 'pad_token') else None

    # Initialize the probability matrix with mlm_probability
    probability_matrix = np.full(labels.shape, mlm_probability)
    
    # Manually create a special tokens mask
    special_tokens_mask = np.array(
        [[tok in [cls_token_id, sep_token_id, pad_token_id] for tok in val] for val in labels.tolist()],
        dtype=bool
    )
    
    # Apply the special tokens mask
    probability_matrix[special_tokens_mask] = 0.0
    if pad_token_id is not None:
        padding_mask = labels == pad_token_id
        probability_matrix[padding_mask] = 0.0
    
    # Generate masked indices
    masked_indices = np.random.rand(*labels.shape) < probability_matrix
    labels[~masked_indices] = -100  # Set labels for unmasked tokens to -100

    # Mask 80% of the time
    indices_replaced = (np.random.rand(*labels.shape) < 0.8) & masked_indices
    inputs['input_ids'][indices_replaced] = tokenizer.token_to_id(tokenizer.mask_token)
    
    # Replace 10% of the time with random word
    indices_random = (np.random.rand(*labels.shape) < 0.1) & masked_indices & ~indices_replaced
    random_words = np.random.randint(0, tokenizer.get_vocab_size(), size=labels.shape)
    inputs['input_ids'][indices_random] = random_words[indices_random]

    return inputs, labels

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

    # Load data
    with open("./eo.txt/eo_test.txt", "r", encoding="utf-8") as f:
        texts = f.readlines()
        texts = [text.strip() for text in texts if text.strip()]

    dataset = MLM_Dataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(model_name) # for mlx this needs to be changed to the bert port
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            batch_inputs, labels = mask_tokens(batch, tokenizer, mlm_probability)
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            labels = labels.to(device)

            outputs = model(**batch_inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())