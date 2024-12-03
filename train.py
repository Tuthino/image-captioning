from model import *
from dataset import *
from torch import nn
from config import max_tokens
import torch
   # Download vocabulary from S3 and cache.

dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=50)

train_set, val_set, test_set = split_dataset(dataset)
train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set)

model = ResnetGru()
optimizer = torch.optim.Adam(model.parameters()) # TODO: Fix Resnet Weights
loss_fn = nn.CrossEntropyLoss()

epochs = 1000

for e in range(epochs):
    
    optimizer.zero_grad()
    
    loss_collector = []

    for labels, images in iter(train_loader):

        prediction_logits = model.forward(images)

        labels_indices = torch.Tensor([tokenizer.encode(lab, max_length=max_tokens, padding="max_length", padding_side="right") for lab in labels]).type(torch.LongTensor)
        
        label_logits = nn.functional.one_hot(labels_indices).type(torch.FloatTensor).to(device)
        
        # @TODO Remove training on all predicted tokens after the <stop> token in the labels
        # This should be done in a way that keeps the gradients of the relevant labels in front
    
        loss = loss_fn(label_logits, prediction_logits)
        
        optimizer.step()
        
        loss_collector.append(loss)
        
    print(e, f"{torch.mean(torch.stack(loss_collector))}")