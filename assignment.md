# Assignment

## Instructions

In this assignment, you will apply the computer vision concepts covered in the lesson to perform image classification using the Fashion MNIST dataset. The Fashion MNIST dataset consists of 60,000 28x28 grayscale images of 10 fashion categories.

### Task: Build an Image Classifier

1. Use the provided starter code to load and explore the Fashion MNIST dataset
2. Preprocess the images using appropriate techniques (e.g., normalization, data augmentation)
3. Build a CNN model to classify the images into one of the 10 classes
4. Train your model and evaluate its performance
5. Experiment with at least one technique to improve model performance (e.g., batch normalization, different pooling strategies, additional convolutional layers)
6. Visualize and analyze your results

### Starter Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Classes in Fashion MNIST
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# TODO: Build your CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # TODO: Define your model architecture here

    def forward(self, x):
        # TODO: Define the forward pass here
        return x

# TODO: Train and evaluate your model

# TODO: Visualize results
```

## Submission

- Submit the URL of the GitHub Repository that contains your work to NTU black board.
- Should you reference the work of your classmate(s) or online resources, give them credit by adding either the name of your classmate or URL.
