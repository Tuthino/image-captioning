import torch
from transformers import LlamaTokenizerFast
from dataset import *
from torch import nn
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.add_special_tokens({"pad_token":"<pad>"})

num_tokens = tokenizer.vocab_size + 1

dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=None)

dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=None)
train_set, val_set, test_set = split_dataset(dataset)

def update_set(this_set, text):
    if isinstance(text, str):  
        token_data = tokenizer.encode(text, add_special_tokens=False)
        tokens = set(tokenizer.decode(number) for number in token_data)
        if tokens:
            this_set.update(tokens)
    return this_set

tokens_train = set({})
for text, _ in train_set: update_set(tokens_train, text)
print(tokens_train)

token2idx = {token: idx for idx, token in enumerate(tokens_train)}
token_tree = {i: {} for i in range(10,-1, -1)}

for token, idx in token2idx.items():
        
    try:

        token_tree[len(token)][token] = idx
        
    except:
        print(len(token))


num_tokens = len(token2idx)
idx2token = {k: v for k, v in enumerate(token2idx)}

max_text_length = 200

def indices2text(indices):
    
    text = ""
    
    end_index = -1 #token2idx["<EOS>"]
    #indices = torch.argmax(indices, dim=1)
    
    for idx in indices:
        
        if idx is end_index: break
        
        text += idx2token[int(idx)]
        
    return text

def text2indices(text):
    
    indices = []
    
    text = text.replace(" ", "")
    
    while len(text) > 0:
        
        found = False
    
        for length, sub_tree in token_tree.items():
            
            sub_part = text[:length]
            
            if sub_part in sub_tree:
                
                indices.append(sub_tree[sub_part])
                print(length, indices[-1], text, sub_part)
            
                text = text[length:]
                found = True
  
                break
            
        if not found:
            text = text[1:]
            indices.append(-1)
        

    return torch.stack(nn.functional.one_hot(torch.Tensor(indices).type(torch.LongTensor)))
   
if __name__ == '__main__':
    
    print(indices2text(torch.Tensor([2,4,3,5,6,])))
    print(text2indices("Paneer With Burst Cherry"))
