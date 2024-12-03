from transformers import ResNetModel
from torch import nn
import torch
import numpy as np
from tokenizer import tokenizer, num_tokens
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResnetGru(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18')
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, num_tokens)
        self.embed = nn.Embedding(num_tokens, 512)
        
        self.to(device)
        
    def forward(self, batch_images):
        
        batch_images = batch_images.to(device)
        
        batch_size = batch_images.shape[0]
        
        # Feature Embedding with ResNet18
        
        features = self.resnet(batch_images).pooler_output.squeeze(dim=(2,3)) # (batch_size, 512)
        
        # Embed Starting Token
        
        start_token_id = torch.tensor(tokenizer.bos_token_id).to(device)
        start_token_embedding = self.embed(start_token_id).repeat(batch_size, 1).to(device)
        
        # Run Gru
        
        input = start_token_embedding.unsqueeze(dim=0) # (sequence_length=1, batch_size, 512)
        hidden_state = features.unsqueeze(dim=0) # (1, batch_size, 512)
        
        for t in range(max_tokens - 1):
            output, hidden_state = self.gru(input, hidden_state)
            
            # Use previously generated text as input for future generation
            input = torch.cat((input, output[-1:]), dim=0)
        
        input = input.permute(1, 0, 2) # batch, seq, 512
        logits = self.proj(input) # batch, seq, token_length
        logits = nn.functional.softmax(logits, dim=2)
        
        return logits