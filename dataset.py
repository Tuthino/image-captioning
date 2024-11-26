import os
import pandas as pd
from PIL import Image
from pathlib import Path
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
            
        if max and len(simple_dataset[0]) >= max: break 

    print(f"Number of successfully loaded images: {successfully_loaded}")
    
    return simple_dataset

if __name__ == '__main__':

    dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=10)
