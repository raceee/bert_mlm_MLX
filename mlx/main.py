import mlx.core as mx
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
import mlx.nn as nn
import time

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
        random_words = mx.random.randint(low=0, high=vocab_size, shape=labels.shape) # some nosie
        inputs['input_ids'] = mx.where(random_indices, random_words, inputs['input_ids'])

        return inputs, labels
    
    if isinstance(inputs, list):
        return [mask_tokens_helper(inp, tokenizer, mlm_probability) for inp in inputs]
    else:
        return mask_tokens_helper(inputs, tokenizer, mlm_probability)

def load_tokenizer():
    tokenizer = Tokenizer.from_file("eo_tokenizer.json") # huggingface tokenizers
    return tokenizer


import numpy as np

def mlm_loss_fn(predictions, labels):
    predictions_np = np.array(predictions)
    labels_np = np.array(labels)

    valid_labels_mask = labels_np != -100

    filtered_predictions = predictions_np[valid_labels_mask]
    filtered_labels = labels_np[valid_labels_mask]

    filtered_predictions_mx = mx.array(filtered_predictions)
    filtered_labels_mx = mx.array(filtered_labels)

    # Compute and return the loss
    return mx.mean(nn.losses.cross_entropy(filtered_predictions_mx, filtered_labels_mx))


# def mlm_eval_fn(predictions, labels):
#     # Convert MLX arrays to NumPy for manipulation
#     predictions_np = np.array(predictions)
#     labels_np = np.array(labels)

#     valid_labels_mask = labels_np != -100

#     filtered_predictions = np.argmax(predictions_np, axis=-1)[valid_labels_mask]
#     filtered_labels = labels_np[valid_labels_mask]

#     correct_predictions = (filtered_predictions == filtered_labels).sum()
#     total_predictions = valid_labels_mask.sum()

#     accuracy = correct_predictions / total_predictions # convert back to mlx array
#     return accuracy


def train_one_epoch(model, data_loader, optimizer):
    def compute_loss():
        predictions, _ = model(input_ids, token_type_ids, attention_masks)
        return mlm_loss_fn(predictions, labels)

    for batch in data_loader:
        out = mask_tokens(batch, tokenizer, mlm_probability)
        input_ids = mx.stack([item[0]['input_ids'] for item in out], axis=0).squeeze(axis=1)
        attention_masks = mx.stack([item[0]['attention_mask'] for item in out], axis=0).squeeze(axis=1)
        labels = mx.stack([item[1] for item in out], axis=0).squeeze(axis=1)
        token_type_ids = mx.zeros(input_ids.shape, dtype=mx.int32)  # Assuming token_type_ids are not used
        print("starting to compute loss")
        loss_and_grads_fn = nn.value_and_grad(model, compute_loss)
        loss, grads = loss_and_grads_fn()
        # Update model parameters
        print("Updating model")
        optimizer.update(model.trainable_parameters(), grads)


if __name__ == "__main__":
    # Configuration
    model_name = 'bert-base-uncased'
    max_length = 128
    mlm_probability = 0.15
    batch_size = 16
    epochs = 100
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
    loss_and_grad_fn = nn.value_and_grad(model, mlm_loss_fn)
    optimizer = AdamW(learning_rate=5e-5)
    start = time.time()
    for epoch in range(epochs):
        train_one_epoch(model, dataloader, optimizer)
        print("Epoch", epoch + 1, "done")
    end = time.time()
    print("Time taken:", end - start)