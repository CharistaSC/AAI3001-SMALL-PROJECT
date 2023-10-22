import os
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from torchvision import models, transforms
import time
import matplotlib.pyplot as plt
import cv2

# Load and prepare data
def load_data(data_path, num_images_per_class):
    image_paths = []
    labels = []
    class_folders = glob(os.path.join(data_path, '*'))

    for class_folder in class_folders:
        class_name = os.path.basename(class_folder)
        class_label = class_folders.index(class_folder)

        # Glob all image files within the class folder
        class_images = glob(os.path.join(class_folder, '*.jpg'))  # Adjust file extension based on your image format

        # Limit the number of images per class if needed
        if num_images_per_class > 0:
            class_images = class_images[:num_images_per_class]

        # Append image paths and labels
        image_paths.extend(class_images)
        labels.extend([class_label] * len(class_images))

    return image_paths, labels

data_path = 'EuroSAT_RGB'
num_images_per_class = 2500
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
image_paths, labels = load_data(data_path, num_images_per_class)

# Modify labels as specified
for i in range(len(labels)):
    if labels[i] == class_names.index('PermanentCrop'):
        labels[i] = 1
    if labels[i] == class_names.index('AnnualCrop'):
        labels[i] = 0
    if labels[i] == class_names.index('Forest'):
        labels[i] = 0
    if labels[i] == class_names.index('HerbaceousVegetation'):
        labels[i] = 1

# Split the dataset into train, validation, and test sets
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=42)

# Custom dataset class with on-the-fly loading and resizing
class EuroSATDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, num_classes=10):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)[:, :, ::-1]  # Read image in BGR format
        image = cv2.resize(image, (128, 128))  # Resize the image

        if self.transform:
            image = self.transform(image)

        # Create a tensor with 0s and 1s
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        class_labels = self.labels[index]
        label[class_labels] = 1

        return image, label

# Define data transformations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Choose the desired data augmentation scheme (horizontal flip in this case)
val_transform_horizontal_flip = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Create datasets and dataloaders
train_dataset = EuroSATDataset(train_paths, train_labels, transform=transform)
val_dataset = EuroSATDataset(val_paths, val_labels, transform=transform)
test_dataset = EuroSATDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Modify the final layer for multi-label classification
num_classes = 10  # Number of classes (labels)
model.classifier[1] = nn.Sequential(
    nn.Linear(1280, num_classes),
    nn.Sigmoid()  # Use sigmoid activation for multi-label classification
)

# Loss and optimizer for multi-label classification
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Adjust parameters as needed

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(12):
    print("\n")

print("Model used: mobilenet_v2 \n")

print(f"Using device: {device} \n")

# Move the model to the GPU if available
model.to(device)

train_losses = []
val_losses = []

print("Start of model training...\n")

# Set to False if you don't want data augmentation during validation
perform_data_augmentation = True
print(f'Data Augmentation set to: {perform_data_augmentation} \n')

# Training loop
num_epochs = 8
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        # Modify labels for multi-label classification as specified
        labels_modified = labels.clone()
        labels_modified[labels_modified == class_names.index('PermanentCrop')] = class_names.index('AnnualCrop')
        labels_modified[labels_modified == class_names.index('AnnualCrop')] = class_names.index('PermanentCrop')
        labels_modified[labels_modified == class_names.index('Forest')] = class_names.index('HerbaceousVegetation')

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate average training loss for the epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

   # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to the same device as the model
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            # Modify labels for multi-label classification as specified
            labels_modified = labels.clone()
            labels_modified[labels_modified == class_names.index('PermanentCrop')] = class_names.index('AnnualCrop')
            labels_modified[labels_modified == class_names.index('AnnualCrop')] = class_names.index('PermanentCrop')
            labels_modified[labels_modified == class_names.index('Forest')] = class_names.index('HerbaceousVegetation')
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Move the outputs to the CPU and then convert to NumPy
            #outputs_cpu = outputs.cpu().numpy()
            #val_predictions.append(outputs_cpu)
            
    # Calculate average validation loss for the epoch
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Print and store losses
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {int(epoch_time // 60)} mins {int(epoch_time % 60)} seconds')

if perform_data_augmentation:
    # Validation with different data augmentations
    val_transforms = [val_transform_horizontal_flip]

    for idx, val_transform in enumerate(val_transforms):
        val_dataset_augmented = EuroSATDataset(val_paths, val_labels, transform=val_transform)
        val_loader_augmented = DataLoader(val_dataset_augmented, batch_size=16)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader_augmented:
                # Move data to the same device as the model
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                # Modify labels for multi-label classification as specified
                labels_modified = labels.clone()
                labels_modified[labels_modified == class_names.index('PermanentCrop')] = class_names.index('AnnualCrop')
                labels_modified[labels_modified == class_names.index('AnnualCrop')] = class_names.index('PermanentCrop')
                labels_modified[labels_modified == class_names.index('Forest')] = class_names.index('HerbaceousVegetation')

                loss = criterion(outputs, labels_modified)
                val_loss += loss.item()

        # Print average validation loss after augmentation
        avg_val_loss = val_loss / len(val_loader_augmented)
        print(f'Validation with Data Augmentation {idx + 1}, Loss: {avg_val_loss:.4f}')
        print("\n")

# Evaluate on the test set
print("Start of model evaluation...\n")
model.eval()
with torch.no_grad():
    test_loss = 0.0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Modify labels for multi-label classification as specified
        labels_modified = labels.clone()
        labels_modified[labels_modified == class_names.index('PermanentCrop')] = class_names.index('AnnualCrop')
        labels_modified[labels_modified == class_names.index('AnnualCrop')] = class_names.index('PermanentCrop')
        labels_modified[labels_modified == class_names.index('Forest')] = class_names.index('HerbaceousVegetation')

        loss = criterion(outputs, labels)
        test_loss += loss.item()

# Calculate average test loss
avg_test_loss = test_loss / len(test_loader)

# Print test metrics
print(f'Test Loss: {avg_test_loss:.4f}')

# Plot the curves after training
plt.figure(figsize=(12, 4))

# Plot Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
