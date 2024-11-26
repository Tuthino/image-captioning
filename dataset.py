import os
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import *

def create_simple_dataset(csv_path, image_folder, size=(224, 224), max=None):
    data = pd.read_csv(csv_path)

    simple_dataset = [[], []]  
    successfully_loaded = 0

    for _, row in data.iterrows():
        image_path = Path(image_folder) / f"{row['Image_Name']}.jpg"
        try:
            image = Image.open(image_path)
            image.thumbnail(size, Image.Resampling.LANCZOS)
            simple_dataset[0].append(row['Title'])
            simple_dataset[1].append(image)
            successfully_loaded += 1
        except Exception as e:
            print(f"Failed to load image: {image_path} | Error: {e}")
            simple_dataset[0].append(row['Title'])
            simple_dataset[1].append(None)
            
        if max and len(simple_dataset[0]) >= max: 
            break 

    print(f"Number of successfully loaded images: {successfully_loaded}")
    
    return simple_dataset

def split_dataset(simple_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    titles = simple_dataset[0]
    images = simple_dataset[1]

    combined = list(zip(titles, images))

    train_data, temp_data = train_test_split(combined, train_size=train_ratio, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    return train_data, val_data, test_data

if __name__ == '__main__':
    dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=10)

    train_set, val_set, test_set = split_dataset(dataset)

    print(f"Train size: {len(train_set)}")
    print(f"Validation size: {len(val_set)}")
    print(f"Test size: {len(test_set)}")
