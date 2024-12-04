import torch
from transformers import LlamaTokenizerFast
from dataset import *

# Initialize tokenizer
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.add_special_tokens({"pad_token": "<pad>"})  # Add a pad token 


dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=None)
train_set, val_set, test_set = split_dataset(dataset)

def update_set(this_set, text):
    if isinstance(text, str):  
        token_data = tokenizer.encode(text, add_special_tokens=False)
        tokens = set(tokenizer.decode(number) for number in token_data)
        if tokens:
            this_set.update(tokens)
    return this_set

tokens_train = set()
for text, _ in train_set:
    tokens_train = update_set(tokens_train, text)

tokenizer.add_tokens(list(tokens_train))

print(f"Initial vocabulary size: {tokenizer.vocab_size}")
print(f"New vocabulary size: {tokenizer.vocab_size + len(tokens_train)}")

# Example 
for text, _ in train_set[:3]:  
    print(f"Original Text: {text}")
    print(f"Tokenized: {tokenizer.encode(text)}")
    print(f"Decoded: {tokenizer.decode(tokenizer.encode(text))}")

