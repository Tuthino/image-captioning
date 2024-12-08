# dataset.py

import os
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

class SimpleDataset(Dataset):
    def __init__(self, data, transform=None):
        self.titles = [row[0] for row in data]
        self.images = [row[1] for row in data]
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        title = self.titles[idx]
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return title, image

def create_simple_dataset(csv_path, image_folder, size=(224, 224), max=None):
    data = pd.read_csv(csv_path)

    simple_dataset = []
    
    transform = transforms.Resize(size)  # Only resize; do not convert to tensor here

    for _, row in data.iterrows():
        image_path = Path(image_folder) / f"{row['Image_Name']}.jpg"
        try:
            with Image.open(image_path).convert("RGB") as image:
                # Apply transformation (resize)
                image = transform(image)
                simple_dataset.append((row['Title'], image))
        except Exception as e:
            print(f"Failed to load image: {image_path} | Error: {e}")

        if max and len(simple_dataset) >= max:
            break

    print(f"Number of successfully loaded images: {len(simple_dataset)}")
    return simple_dataset

def split_dataset(simple_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    seed = 42
    train_data, temp_data = train_test_split(simple_dataset, train_size=train_ratio, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)
    return train_data, val_data, test_data

def collate_fn(batch):
    """
    Custom collate function to handle batches of (caption, image) tuples.
    """
    captions, images = zip(*batch)
    return list(captions), list(images)



def data_loader(train_set, val_set, test_set, batch_size=30):
    train_dataset = SimpleDataset(train_set)
    val_dataset = SimpleDataset(val_set)
    test_dataset = SimpleDataset(test_set)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn )

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example usage
    csv_path = "path/to/your/captions.csv"        # Replace with your CSV path
    image_folder = "path/to/your/images"         # Replace with your image folder path
    dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=100)

    train_set, val_set, test_set = split_dataset(dataset)
    train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set, batch_size=30)
    
    train_titles, train_images = next(iter(train_loader))
    print(train_titles)
    print(f"Images batch shape: {train_images.size()}")
    # plt.imshow(train_images[0].permute(1, 2, 0))
    # plt.title(train_titles[0])
    # plt.axis('off')
    # plt.show()
