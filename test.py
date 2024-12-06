from model import *
from dataset import *
from tokenizer import *

tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
tokenizer.add_special_tokens({"pad_token": "<pad>"})
num_tokens = tokenizer.vocab_size + 1

dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=None)


train_set, val_set, test_set = split_dataset(dataset)
train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set)

token_tree, token2idx, idx2token, num_tokens = init_tokenizer_set(train_set, tokenizer)
model = ResnetGru(num_tokens)

for titles, images in iter(train_loader):

    result = model.forward(images, tokenizer, max_tokens)

    # apply argmax to get the most likely token
    token_indices = torch.argmax(result, dim=-1)

    print(result.shape, result)

    print(token_indices)
    for indices, title in zip(token_indices, titles):
        print(title, indices)
        print(title, indices2text(indices, idx2token))
        # print(title, indices)
