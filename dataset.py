import os
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from config import *


class SimpleDataset(Dataset):
    def __init__(self, data):
        
        # Resize images to common shape and convert PIL -> Tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
        
        self.titles = [row[0] for row in data]
        self.images = torch.stack([transform(row[1]) for row in data])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        title = self.titles[idx]
        image = self.images[idx]
        
        return title, image

def create_simple_dataset(csv_path, image_folder, size=(224, 224), max=None):
    data = pd.read_csv(csv_path)

    simple_dataset = []
    successfully_loaded = 0

    for _, row in data.iterrows():
        image_path = Path(image_folder) / f"{row['Image_Name']}.jpg"
        try:
            image = Image.open(image_path)
            simple_dataset.append((row['Title'], image))
            successfully_loaded += 1
        except Exception as e:
            print(f"Failed to load image: {image_path} | Error: {e}")
            simple_dataset.append((row['Title'], None))

        if max and len(simple_dataset) >= max:
            break

    print(f"Number of successfully loaded images: {successfully_loaded}")

    return simple_dataset

def split_dataset(simple_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    seed = 42
    
    train_data, temp_data = train_test_split(simple_dataset, train_size=train_ratio, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)
    return train_data, val_data, test_data

def data_loader(train_set, val_set, test_set):
    train_dataset = SimpleDataset(train_set)
    val_dataset = SimpleDataset(val_set)
    test_dataset = SimpleDataset(test_set)

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':

    dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=10)

    train_set, val_set, test_set = split_dataset(dataset)
    train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set)
    
    train_titles, train_images = next(iter(train_loader))
    print(train_titles)
    print(f"Images batch shape: {train_images.size()}")
    plt.imshow(train_images[0].permute(1, 2, 0))
    plt.show()
