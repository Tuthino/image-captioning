# train.py
from config import *
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import (
    AutoFeatureExtractor,
    VisionEncoderDecoderModel,
    BartTokenizer,
    LlamaTokenizer,
    BartForConditionalGeneration,
    AdamW,
    pipeline
)
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sacrebleu

from dataset import create_simple_dataset, split_dataset, data_loader

# Configuration variables (replace with your actual paths or values)
# csv_path = "path/to/your/captions.csv"        # Replace with your CSV path
# image_folder = "path/to/your/images"         # Replace with your image folder path
# max_tokens = 50
# resnet_model_save_path = "best_model.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

########################################
# Step 1: Load Your Dataset
########################################
# dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=20)
# train_set, val_set, test_set = split_dataset(dataset)
# train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set, batch_size=20)

########################################
# Step 2: Model and Tokenizer Setup
########################################
def setup_model_and_tokenizer():
    # Use a publicly available BART model as decoder
    encoder_name = "google/vit-base-patch16-224-in21k"
    decoder_name = "facebook/bart-base"
    
    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
    
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(decoder_name)

    
    # Load VisionEncoderDecoderModel
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_name,
        decoder_name
    )
    
    # Set special tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    # print(f"Decoder vocab size: {model.config.vocab_size}")
    # print(f"Max token index in input_ids: {input_ids.max().item()}")

    # Optionally set generation parameters
    model.config.max_length = max_tokens
    model.config.num_beams = 5
    
    # Move to device
    model.to(device)
    
    return model, tokenizer, feature_extractor

########################################
# Step 3: BLEU Score Evaluation
########################################
def compute_bleu(predictions, references):
    # sacrebleu expects a list of predicted strings and a list of reference strings
    # references should be a list of reference lists for each prediction
    # For a single reference per prediction, wrap references in a list
    return sacrebleu.corpus_bleu(predictions, [references]).score

def indices2text(indices, tokenizer):
    # Convert list of token IDs to text
    text = tokenizer.decode(indices, skip_special_tokens=True)
    return text

@torch.no_grad()
def evaluate_bleu(model, val_loader, tokenizer, feature_extractor, device):
    model.eval()
    predictions = []
    references = []

    for batch in val_loader:
        titles, images = batch
        images = images
        
        # Collect references
        references.extend(titles)
        
        # Feature extraction
        # The feature extractor expects images as a list of PIL images or tensors
        # Since images are PIL Images, use them directly
        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)

        # Generate captions
        generated_ids = model.generate(pixel_values, max_length=max_tokens, num_beams=5)
        
        # Decode generated captions
        generated_captions = [indices2text(generated_id, tokenizer) for generated_id in generated_ids]
        
        # Collect predictions
        predictions.extend(generated_captions)
    
    # Compute BLEU score
    bleu_score = compute_bleu(predictions, references)
    return bleu_score

########################################
# Step 4: Training Loop
########################################
def train_model(train_loader, val_loader, model, optimizer, tokenizer, feature_extractor, device):
    best_val_bleu = -float('inf')
    patience = 50
    patience_counter = 0
    epochs = 200

    for epoch in range(epochs):
        model.train()
        loss_collector = []
        
        for batch_idx, (titles, images) in enumerate(train_loader):
            # images is a list of PIL Images
            # titles is a list of strings

            # Tokenize captions
            encoding = tokenizer(
                list(titles),
                padding="longest",
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # Prepare labels (input_ids shifted for teacher forcing)
            labels = input_ids.clone()
            
            # Feature extraction for the encoder (ViT)
            pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
            
            # Forward pass: ensure correct arguments are passed
            outputs = model(
                pixel_values=pixel_values,  # For the encoder
                # input_ids=input_ids,        # For the decoder
                # attention_mask=attention_mask,
                labels=labels               # For loss calculation
            )
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            loss_collector.append(loss.item())
        
        avg_train_loss = sum(loss_collector) / len(loss_collector)
        
        # Evaluate BLEU on validation set
        val_bleu = evaluate_bleu(model, val_loader, tokenizer, feature_extractor, device)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val BLEU: {val_bleu:.2f}")
        
        # Early stopping based on BLEU score
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            patience_counter = 0
            torch.save(model.state_dict(), best_model_save_path)
            print(f"New best model saved with BLEU score: {val_bleu:.2f}")
        else:
            patience_counter += 1
            print(f"No improvement in BLEU for {patience_counter} epochs.")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

########################################
# Step 5: Main Execution
########################################
if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # Load dataset
    dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=10)
    train_set, val_set, test_set = split_dataset(dataset)
    train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set, batch_size=10)
    
    # Print a sample batch
    train_titles, train_images = next(iter(train_loader))
    print("Sample Titles:", train_titles)
    # print(f"Images batch shape: {train_images.size()}")
    # plt.imshow(train_images[0].permute(1, 2, 0).cpu().numpy())
    # plt.title(train_titles[0])
    # plt.axis('off')
    # plt.show()
    
    # Setup model and tokenizer
    model, tokenizer, feature_extractor = setup_model_and_tokenizer()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Start training
    train_model(train_loader, val_loader, model, optimizer, tokenizer, feature_extractor, device)
    
    # Load the best model
    model.load_state_dict(torch.load(best_model_save_path))
    model.to(device)
    model.eval()
    
    # Create an inference pipeline
    caption_pipeline = pipeline(
        "image-to-text",
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        # device=0 if torch.cuda.is_available() else -1
        device=device
    )
    
    # Test the pipeline with a sample image
    captions = caption_pipeline(test_image_path)
    print(f"Generated Caption: {captions[0]['generated_text']}")
