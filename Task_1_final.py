import os
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
from torchvision.transforms.functional import to_tensor
from torchvision import models, transforms
import time
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

for i in range(12):
    print("\n")

def load_data(data_path):
    image_paths = []
    labels = []
    class_folders = glob(os.path.join(data_path, '*'))

    for class_folder in class_folders:
        class_name = os.path.basename(class_folder)
        class_label = class_folders.index(class_folder)

        # Glob all image files within the class folder
        class_images = glob(os.path.join(class_folder, '*.jpg'))  # Adjust file extension based on your image format

        # Append image paths and labels
        image_paths.extend(class_images)
        labels.extend([class_label] * len(class_images))

    return image_paths, labels

data_path = 'EuroSAT_RGB'
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
image_paths, labels = load_data(data_path)

train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=42)

# Custom dataset class with on-the-fly loading and resizing
class EuroSATDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)[:, :, ::-1]  # Read image in BGR format
        image = cv2.resize(image, (224, 224))  # Resize the image

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

# Define data transformations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform_horizontal_flip = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform_color_jitter = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform_random_rotation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
model.classifier[1] = nn.Linear(1280, 10)  # 10 Classes

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
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
val_accuracies = []

print("Start of model training...\n")

perform_data_augmentation = False
perform_class_validation = False
print(f'Validate Data Augmentation set to: {perform_data_augmentation} \n')
print(f'Class Validation set to: {perform_class_validation} \n')

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    train_loss = 0.0
    # Set to False if you don't want data augmentation during validation

    for images, labels in train_loader:
        # Move data to the same device as the model
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
    correct = 0
    total = 0
    val_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to the same device as the model
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store predictions for later average precision computation
            val_predictions.append(outputs.cpu().numpy())

    # Calculate average validation loss for the epoch
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Calculate accuracy
    accuracy = correct / total
    val_accuracies.append(accuracy)

    # Calculate average precision for each class on the validation set
    val_predictions = np.concatenate(val_predictions, axis=0)
    average_precision_val = average_precision_score(val_labels, val_predictions, average=None)

    # Print and store losses, accuracy, and average precision
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}, Time: {int(epoch_time // 60)} mins {int(epoch_time % 60)} seconds')

    if perform_class_validation:
        # Print average precision for each class on the validation set
        for i, ap in enumerate(average_precision_val):
            print(f'Class {i}: Average Precision: {ap:.4f}')
        # Calculate and print mean average precision
        mean_average_precision_val = np.mean(average_precision_val)
        print(f'Mean Average Precision: {mean_average_precision_val:.4f} \n')

    if perform_data_augmentation:
        # Validation with different data augmentations
        val_transforms = [val_transform_horizontal_flip, val_transform_color_jitter, val_transform_random_rotation]

        for idx, val_transform in enumerate(val_transforms):

            val_dataset_augmented = EuroSATDataset(val_paths, val_labels, transform=val_transform)
            val_loader_augmented = DataLoader(val_dataset_augmented, batch_size=16)

            model.eval()
            correct = 0
            total = 0
            val_predictions = []

            with torch.no_grad():
                for images, labels in val_loader_augmented:
                    # Move data to the same device as the model
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # Store predictions for later average precision computation
                    val_predictions.append(outputs.cpu().numpy())

            # Calculate average precision for each class on the augmented validation set
            val_predictions = np.concatenate(val_predictions, axis=0)
            average_precision_augmented_val = average_precision_score(val_labels, val_predictions, average=None)

            # Print average precision for each class after augmentation
            for i, ap in enumerate(average_precision_augmented_val):
                print(f'Validation with Data Augmentation {idx + 1}, Class {i}, Average Precision: {ap:.4f}')
            print("\n")

# Evaluate on the test set
print("Start of model evaluation...\n")
model.eval()
with torch.no_grad():
    test_loss = 0.0
    correct = 0
    total = 0
    all_test_predictions = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Save predictions for further analysis if needed
        all_test_predictions.append(outputs.cpu().numpy())

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)

    # Concatenate all predictions outside the loop
    test_predictions = np.concatenate(all_test_predictions, axis=0)

# Calculate average precision for each class on the test set
average_precision_test_per_class = average_precision_score(test_labels, test_predictions, average=None)
average_precision_test = average_precision_score(test_labels, test_predictions, average='macro')
accuracy_test = accuracy_score(test_labels, test_predictions.argmax(axis=1))
mean_average_precision_test = np.mean(average_precision_test_per_class)

# Print test metrics
print(f'Test Loss: {avg_test_loss:.4f}')
print(f'Test Accuracy: {accuracy_test:.4f}')
print(f'Test Mean Average Precision: {mean_average_precision_test:.4f}')

if perform_class_validation:
    # Print average precision for each class on the test set
    for class_idx, ap in enumerate(average_precision_test_per_class):
        print(f'Test Class {class_idx}: Average Precision: {ap:.4f}')

# Plot the curves after training
plt.figure(figsize=(12, 4))

# Plot Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
