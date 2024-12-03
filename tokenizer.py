import torch
from transformers import LlamaTokenizerFast, LlamaTokenizer
from dataset import *
#tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
#import pdb; pdb.set_trace()
tokenizer.add_special_tokens({"pad_token":"<pad>"})

num_tokens = tokenizer.vocab_size + 1

#from tokenizer import *

dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=50)

train_set, val_set, test_set = split_dataset(dataset)

def update_set(this_set, text):
    
    if type(text) == float: return this_set
    
    token_data = tokenizer.encode(text)
    tokens = set(tokenizer.decode(number) for number in token_data)
    
    if tokens: return this_set.update(tokens)
    
    return this_set

tokens_train = set({})
for text, _ in train_set:
    update_set(tokens_train, text)
print(tokens_train)

tokenizer = LlamaTokenizer(tokenizer_object=None)
tokenizer.add_tokens(tokens_train)
vocab = {token: idx for idx, token in enumerate(tokens_train)}
tokenizer.vocab = vocab

pass
# characters = ['<SOS>', '<EOS>', '<PAD>', '     ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# num_tokens = len(characters)
# idx2char = {k: v for k, v in enumerate(characters)}
# char2idx = {v: k for k, v in enumerate(characters)}

# max_text_length = 200

# def indices2text(indices):
    
#     text = ""
    
#     end_index = char2idx["<EOS>"]
#     indices = torch.argmax(indices, dim=1)
    
#     for idx in indices:
        
#         if idx is end_index: break
        
#         text += idx2char[int(idx)]
        
#     return text

# def text2indices(text):
    
#     indices = []
    
#     for char in list(text):
        
#         if char not in char2idx: break
        
#         indices.append(char2idx[char])

#        return torch.stack(nn.functional.one_hot(indices))