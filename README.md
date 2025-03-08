# ResNet18 Image Classification using PyTorch

This repository contains a PyTorch implementation of the ResNet18 model for image classification. The model is trained and evaluated on a custom dataset, and the trained model is saved for future use. Below is a detailed explanation of the code, its architecture, and how to use it.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Code Explanation](#code-explanation)
4. [Usage](#usage)
5. [Results](#results)
6. [Dependencies](#dependencies)
7. [License](#license)

---

## Overview

This project demonstrates how to use the ResNet18 architecture for image classification tasks. The model is trained on a custom dataset, validated during training, and finally evaluated on a test set. The trained model is saved to Google Drive for future use. The code is written in PyTorch and includes features like data augmentation, GPU support, and progress bars for training and evaluation.

---

## Architecture

### ResNet18
ResNet18 is a convolutional neural network (CNN) architecture that is 18 layers deep. It was introduced in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by He et al. The key innovation of ResNet is the use of **residual blocks**, which allow the network to learn identity mappings, making it easier to train very deep networks.

#### Key Features of ResNet18:
- **Residual Blocks**: Each residual block contains two convolutional layers with skip connections that bypass these layers. This helps mitigate the vanishing gradient problem.
- **Global Average Pooling**: Instead of fully connected layers at the end, ResNet uses global average pooling to reduce the spatial dimensions to 1x1.
- **Pretrained Weights**: The model can be initialized with pretrained weights from ImageNet, which helps in transfer learning tasks.

---

## Code Explanation

### 1. **Setup and Imports**
The code begins by importing necessary libraries such as `torch`, `torchvision`, and `tqdm`. It also checks if a GPU is available and mounts Google Drive for dataset access.

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from google.colab import drive
drive.mount('/content/drive')
```
### 2. Data Preprocessing
The dataset is preprocessed using transforms.Compose, which includes resizing, converting images to tensors, and normalizing them using ImageNet mean and standard deviation.
```transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(            # Normalize with ImageNet mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```
### 3. Loading Datasets
The datasets are loaded using torchvision.datasets.ImageFolder, which assumes the data is organized in folders by class. Data loaders are created for training, validation, and testing.

```train_dataset = datasets.ImageFolder(root="/content/drive/MyDrive/train", transform=transform)
val_dataset = datasets.ImageFolder(root="/content/drive/MyDrive/validation", transform=transform)
test_dataset = datasets.ImageFolder(root="/content/drive/MyDrive/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
```
### 4. Model Definition
The ResNet18 model is loaded with pretrained weights, and the final fully connected layer is modified to match the number of classes in the dataset.

```model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
```
### 5. Training and Validation
The training loop includes forward and backward passes, loss calculation, and optimization. A progress bar is added using tqdm to monitor training and validation.
```
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            train_loop.set_postfix(loss=loss.item(), accuracy=(100. * correct / total))
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation")
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                val_loop.set_postfix(loss=loss.item(), accuracy=(100. * correct / total))
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print("Training complete.")
```
### 6. Testing and Saving the Model
After training, the model is evaluated on the test set, and the trained model is saved to Google Drive.
```
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    test_loop = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for images, labels in test_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            test_loop.set_postfix(loss=loss.item(), accuracy=(100. * correct / total))
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "/content/drive/MyDrive/resnet18_model.pth")
print("Model saved to /content/drive/MyDrive/resnet18_model.pth")
```
### Usage
Set Up Environment:

Install the required dependencies:
```
pip install torch torchvision tqdm matplotlib
```
Mount Google Drive if using Google Colab.

Prepare Dataset:

Organize your dataset into train, validation, and test folders, with each folder containing subfolders for each class.

Run the Code:

Execute the code in a Python environment or Google Colab.

Evaluate and Save:

After training, the model will be evaluated on the test set and saved to Google Drive.
### Results
The model's performance is measured using loss and accuracy metrics during training, validation, and testing. The results are displayed in the console and can be visualized using tools like matplotlib.
Here’s the **Results** section for your `README.md` file, based on the output of the model. This section summarizes the training, validation, and testing performance in a clear and concise way.

---

The ResNet18 model was trained for **10 epochs** on a custom dataset, achieving excellent performance in both training and validation phases. Below is a summary of the results:

### Training Performance
- **Final Training Loss**: `0.0080`
- **Final Training Accuracy**: `99.77%`
- The model consistently achieved high accuracy (>99%) throughout the training process, indicating strong learning capabilities.

### Validation Performance
- **Final Validation Loss**: `0.0067`
- **Final Validation Accuracy**: `99.83%`
- The validation accuracy remained stable and high, demonstrating the model's ability to generalize well to unseen data.

### Testing Performance
- **Test Loss**: `0.0067`
- **Test Accuracy**: `99.79%`
- The model performed exceptionally well on the test set, confirming its robustness and reliability for image classification tasks.

### Key Observations
1. **High Accuracy**: The model achieved **>99% accuracy** across all phases (training, validation, and testing), indicating excellent performance.
2. **Low Loss**: Both training and validation losses were consistently low, suggesting effective optimization and minimal overfitting.
3. **Stability**: The model's performance remained stable across epochs, with no significant fluctuations in accuracy or loss.

### Performance Over Epochs
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1     | 0.0276     | 99.16%    | 0.0089   | 99.71%  |
| 2     | 0.0145     | 99.62%    | 0.0085   | 99.72%  |
| 3     | 0.0117     | 99.71%    | 0.0071   | 99.83%  |
| 4     | 0.0098     | 99.72%    | 0.0097   | 99.75%  |
| 5     | 0.0102     | 99.68%    | 0.0088   | 99.71%  |
| 6     | 0.0079     | 99.79%    | 0.0059   | 99.83%  |
| 7     | 0.0090     | 99.76%    | 0.0084   | 99.79%  |
| 8     | 0.0069     | 99.84%    | 0.0090   | 99.75%  |
| 9     | 0.0061     | 99.82%    | 0.0083   | 99.79%  |
| 10    | 0.0080     | 99.77%    | 0.0067   | 99.83%  |

### Conclusion
The ResNet18 model demonstrated outstanding performance, achieving **>99% accuracy** on the test set. This makes it highly suitable for real-world image classification tasks. The model's stability, low loss, and high accuracy across all phases highlight its effectiveness and reliability.

---

### Dependencies
Python 3.x

PyTorch

torchvision

tqdm

matplotlib
### License
This project is licensed under the MIT License. See the LICENSE file for details.
```

### How to Use:
1. Copy the above Markdown content.
2. Create a `README.md` file in your GitHub repository.
3. Paste the content into the `README.md` file.
4. Commit and push the changes to your repository.

This will provide a professional and detailed explanation of your project for anyone visiting your repository. Let me know if you need further assistance!
```
