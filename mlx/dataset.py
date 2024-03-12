import torch
import mlx.core as mx

class MLXDataset:
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.custom_encode_plus(text,
            padding='max_length',
            return_tensors='mx'
        )
        # inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        return inputs
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def custom_encode_plus(self, text, padding='max_length', return_tensors=None):
        def custom_encode_helper(text,padding,return_tensors):
            encoded = self.tokenizer.encode(text)
        
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
            
            else:
                output['input_ids'] = mx.array([output['input_ids']])
                output['attention_mask'] = mx.array([output['attention_mask']])
            
            return output



        # Tokenize input text
        if isinstance(text, list):
            output = [custom_encode_helper(t, padding, return_tensors) for t in text]
            return output
        else:
            return custom_encode_helper(text, padding, return_tensors)