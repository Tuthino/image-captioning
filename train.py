from cProfile import label
from model import *
from dataset import *
from torch import nn
from config import max_tokens
import torch

def train(train_loader, model, optimizer, tokenizer, token_tree, token2idx):
    best_val_loss = float('inf')  # Initialize with infinity
    patience = 40              # Number of epochs to wait for improvement
    patience_counter = 0         # Counter for non-improving epochs
    epochs = 1000

    for e in range(epochs):
        
        optimizer.zero_grad()
        
        loss_collector = []

        for batch in iter(train_loader):
            labels, images = batch
            images = images.to(device)
            # Convert labels to indices
            labels = [text2indices(label, token_tree, token2idx) for label in labels]

            # Pad the sequences to the same length
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=token2idx['<pad>']).to(device)

            # Ensure labels have the correct shape
            prediction_logits = model.forward(images, tokenizer, max_tokens)
            batch_size, seq_len, vocab_size = prediction_logits.shape

            # Check if labels need padding to match seq_len
            if labels.shape[1] != seq_len:
                labels = torch.nn.functional.pad(labels, (0, 0, 0, seq_len - labels.shape[1]), value=token2idx['<pad>'])

            # Flatten the prediction logits and labels
            prediction_logits = prediction_logits.view(batch_size * seq_len, vocab_size)
            # print(f"Prediction logits shape: {prediction_logits.shape}")
            # print(f"Labels shape before view: {labels.shape}")
            # Remove the vocabulary dimension from labels
            labels = labels.argmax(dim=-1)  # Shape: (batch_size, seq_len)

            # Flatten labels to match prediction_logits
            labels = labels.view(batch_size * seq_len)

            loss_fn = nn.CrossEntropyLoss(ignore_index=token2idx['<pad>']) # ignore padding tokens
            loss = loss_fn(prediction_logits, labels)
            loss.backward()
            optimizer.step()

            loss_collector.append(loss.item())
        
        print(f"Epoch {e+1}/{epochs}, Loss: {sum(loss_collector)/len(loss_collector)}")

    torch.save(model.state_dict(), resnet_model_save_path)

if __name__ == '__main__':
    dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224),max=50)
    tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_set, val_set, test_set = split_dataset(dataset)
    train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set)

    token_tree, token2idx, idx2token, num_tokens = init_tokenizer_set(train_set, tokenizer)
    print(num_tokens)
    model = ResnetGru(num_tokens)
    optimizer = torch.optim.Adam(model.parameters()) # TODO: Fix Resnet Weights
    loss_fn = nn.CrossEntropyLoss()
    train(train_loader, model, optimizer, tokenizer, token_tree, token2idx)