import mlx.core as mx
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
    
    def custom_encode_plus(self, padding='max_length', return_tensors=None):
        # Tokenize input text
        encoded = tokenizer.encode(self.text)
        
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
        inputs = self.encode_plus(
            max_length=self.max_length,
            padding='max_length',
            return_tensors='np'
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        return inputs

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling."""
    labels = inputs['input_ids'].clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    padding_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # Mask 80% of the time
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs['input_ids'][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # Replace 10% of the time with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs['input_ids'][indices_random] = random_words[indices_random]

    # The rest of the time (10%) we keep the original tokens unchanged

    return inputs, labels
if __name__ == "__main__":
    # Configuration
    model_name = 'bert-base-uncased'
    max_length = 128
    mlm_probability = 0.15
    batch_size = 16
    epochs = 3
    tokenizer = load_tokenizer()

    # Load data
    with open("./eo.txt/eo_test.txt", "r") as f:
        texts = f.readlines()
        texts = [text.strip() for text in texts if text.strip()]

    dataset = MLM_Dataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            batch_inputs, labels = mask_tokens(batch, tokenizer, mlm_probability)
            print("BATCH INPUTS: ", batch_inputs)
            print("LABELS", labels)
            # batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            # labels = labels.to(device)

            # outputs = model(**batch_inputs, labels=labels)
            # loss = outputs.loss

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # loop.set_description(f'Epoch {epoch}')
            # loop.set_postfix(loss=loss.item())