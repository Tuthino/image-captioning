import torch
from transformers import LlamaTokenizerFast
from dataset import *
from torch import nn

max_text_length = 200


def update_set(this_set, text, tokenizer):
    if isinstance(text, str):
        token_data = tokenizer.encode(text, add_special_tokens=False)
        tokens = set(tokenizer.decode(number) for number in token_data)
        if tokens:
            this_set.update(tokens)
    return this_set


def init_tokenizer_set(train_set, tokenizer):
    tokens_train = set({})
    for text, _ in train_set:
        update_set(tokens_train, text, tokenizer)
    print("tokens train")
    print(tokens_train)

    token2idx = {token: idx for idx, token in enumerate(tokens_train)}
    token_tree = {i: {} for i in range(10, -1, -1)}

    for token, idx in token2idx.items():

        try:

            token_tree[len(token)][token] = idx

        except:
            print(len(token))

    num_tokens = len(token2idx)
    # idx2token = {k: v for k, v in enumerate(token2idx)}
    idx2token = {idx: token for token, idx in token2idx.items()}

    return token_tree, token2idx, idx2token, num_tokens


# def indices2text(indices, idx2token):
#     text = []
#     end_index = -1  # token2idx["<EOS>"]
#     for idx in indices:
#         if idx.numel() == 1:
#             text.append(idx2token[idx.item()])
#         else:
#             text.append([idx2token[i.item()] for i in idx])
#         if idx == end_index:
#             break
#         if idx in idx2token:
#             text += idx2token[int(idx)]
#         else:
#             print(f"Warning: Index {idx} not found in idx2token")
#             text += "<UNK>"  # Add a placeholder for unknown tokens
#     return text
def indices2text(indices, idx2token):
    text = ""
    end_index = -1  # token2idx["<EOS>"]
    # indices = torch.argmax(indices, dim=1)
    for idx in indices:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()  # convert to single-element scalar
        if idx is end_index:
            break
        text += idx2token[int(idx)]
    return text


# def text2indices(text, token_tree):
#     indices = [token_tree.get(token) for token in text.split()]
#     num_classes = len(token_tree)

#     valid_indices = []

#     for idx in indices:
#         if idx is None:
#             print('warning token not found')
#         elif idx < num_classes:
#             valid_indices.append(idx)
#         else:
#             print(f"Warning: Index {idx} not found in token_tree")

#     return torch.stack(
#         [
#             nn.functional.one_hot(torch.tensor(idx), num_classes=num_classes)
#             for idx in valid_indices
#         ]
#     )


def text2indices(text, token_tree, token2idx):
    indices = []
    text = text.replace(" ", "")
    while len(text) > 0:
        found = False
        for length in sorted(token_tree.keys(), reverse=True):
            sub_part = text[:length]
            if sub_part in token_tree[length]:
                indices.append(token_tree[length][sub_part])
                print(length, indices[-1], text, sub_part)
                text = text[length:]
                found = True
                break
        if not found:
            text = text[1:]
            # indices.append(token2idx['<UNK>'])
    
    num_classes = len(token2idx)
    valid_indices = []
    for idx in indices:
        if idx is not None and idx < num_classes:
            valid_indices.append(idx)
        else:
            print(f"Warning: Index {idx} not found in token_tree")

    return torch.stack(
        [
            nn.functional.one_hot(torch.tensor(idx), num_classes=num_classes)
            for idx in valid_indices
        ]
    )
    # return torch.stack(nn.functional.one_hot(torch.Tensor(indices).type(torch.LongTensor)))


if __name__ == "__main__":
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    num_tokens = tokenizer.vocab_size + 1

    dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=None)

    train_set, val_set, test_set = split_dataset(dataset)
    train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set)
    token_tree, token2idx, idx2token, num_tokens = init_tokenizer_set(
        train_set, tokenizer
    )
    print("idx2token test:")
    print(
        indices2text(
            torch.Tensor(
                [
                    2,
                    4,
                    3,
                    5,
                    6,
                ]
            ),
            idx2token,
        )
    )
    print("token tree: ", token_tree)
    print(text2indices("Paneer With Burst Cherry", token_tree, token2idx))
