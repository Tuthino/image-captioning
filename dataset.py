import os
import pandas as pd
import cv2
from config import *

print(os.listdir("."))

data = pd.read_csv(csv_path)

simple_dataset = [[], []]  

for _, row in data.iterrows():    
    image_path= image_folder / (row['Image_Name'] + ".jpg") 
    print(image_path)
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, (224, 224)) / 255.0
        simple_dataset[0].append(row['Title'])
        simple_dataset[1].append(image)
        print("Image loaded")
    else:
        #x+=1
        print(f"Failed to load image: {image_path}")
        simple_dataset[1].append(None) 

print(f"Titles: {simple_dataset[0]}")
#print(x)

# Preprocess images
# def preprocess_image(image_path, target_size=(224, 224)):
#     image = cv2.imread(image_path)
#     if image is None:
#         return None
#     image = cv2.resize(image, target_size) / 255.0  # Normalize to [0, 1]
#     return image

# # Create the vector with image and title
# image_title_vector = []
# for _, row in data.iterrows():
#     image_path = os.path.join(image_folder, row['Image_Name'])
#     image = preprocess_image(image_path)
#     if image is not None:
#         title = row['Title']
#         image_title_vector.append({'image': image, 'title': title})

# Example output
#print(f"First element: {image_title_vector[0]}")
#print(f"Total elements: {len(image_title_vector)}")
