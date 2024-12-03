from model import *
from dataset import *


dataset = create_simple_dataset(csv_path, image_folder, size=(224, 224), max=10)

train_set, val_set, test_set = split_dataset(dataset)
train_loader, val_loader, test_loader = data_loader(train_set, val_set, test_set)

model = ResnetGru()

for titles, images in iter(train_loader):

    result = model.forward(images)

    print(result.shape, result)
    
    for indices, title in zip(result, titles):
        
        print(title, indices2text(indices))