# Assignment Solution

## Instructions

In this assignment, you will apply the computer vision concepts covered in the lesson to perform image classification using the Fashion MNIST dataset. The Fashion MNIST dataset consists of 60,000 28x28 grayscale images of 10 fashion categories.

### Task: Build an Image Classifier

1. Use the provided starter code to load and explore the Fashion MNIST dataset
2. Preprocess the images using appropriate techniques (e.g., normalization, data augmentation)
3. Build a CNN model to classify the images into one of the 10 classes
4. Train your model and evaluate its performance
5. Experiment with at least one technique to improve model performance (e.g., batch normalization, different pooling strategies, additional convolutional layers)
6. Visualize and analyze your results

### Solution Code

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

# Visualize some examples
def show_examples(data_loader, num_samples=5):
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    fig = plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i+1, xticks=[], yticks=[])
        img = images[i].numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{classes[labels[i]]}")
    plt.show()

show_examples(train_loader)

# Define our CNN model
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 3 * 3, out_features=512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Apply first convolutional layer with batch normalization and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Apply second convolutional layer with batch normalization and pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Apply third convolutional layer with batch normalization and pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 3 * 3)

        # Apply fully connected layers with dropout for regularization
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Initialize the model
model = FashionCNN()
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print statistics every 100 mini-batches
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                train_losses.append(running_loss / 100)
                running_loss = 0.0

    print('Finished Training')
    return train_losses

# Train the model
train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=5)

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Mini-batch (x100)')
plt.ylabel('Loss')
plt.show()

# Function to evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate class-wise accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print overall accuracy
    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

    # Print class-wise accuracy
    for i in range(10):
        print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

    return correct / total, class_correct, class_total

# Evaluate the model
accuracy, class_correct, class_total = evaluate_model(model, test_loader)

# Plot the class-wise accuracy
plt.figure(figsize=(12, 6))
plt.bar(range(10), [100 * class_correct[i] / class_total[i] for i in range(10)])
plt.xticks(range(10), classes, rotation=45)
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Class-wise Accuracy')
plt.tight_layout()
plt.show()

# Visualize predictions
def visualize_predictions(model, data_loader, num_samples=10):
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # Get predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Display images and predictions
    fig = plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        ax = fig.add_subplot(2, num_samples//2, i+1, xticks=[], yticks=[])
        img = images[i].numpy().squeeze()
        ax.imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {classes[predicted[i]]}\nTrue: {classes[labels[i]]}", color=color)
    plt.tight_layout()
    plt.show()

# Visualize some predictions
visualize_predictions(model, test_loader)

# Confusion matrix to understand model performance better
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, data_loader):
    # Get all predictions
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(model, test_loader)

# Experiment with different model architectures
class ImprovedFashionCNN(nn.Module):
    def __init__(self):
        super(ImprovedFashionCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

# Initialize the improved model
improved_model = ImprovedFashionCNN()
print(improved_model)

# Train the improved model with the same parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(improved_model.parameters(), lr=0.001)

# Train the model
improved_train_losses = train_model(improved_model, train_loader, criterion, optimizer, num_epochs=5)

# Evaluate the improved model
improved_accuracy, improved_class_correct, improved_class_total = evaluate_model(improved_model, test_loader)

# Compare both models
plt.figure(figsize=(12, 6))
plt.bar(['Original CNN', 'Improved CNN'], [accuracy * 100, improved_accuracy * 100])
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison')
plt.ylim([0, 100])
plt.show()

# Plot confusion matrix for improved model
plot_confusion_matrix(improved_model, test_loader)

# Summary of the assignment results
print("Summary of Results:")
print(f"Original CNN Accuracy: {accuracy * 100:.2f}%")
print(f"Improved CNN Accuracy: {improved_accuracy * 100:.2f}%")
print(f"Improvement: {(improved_accuracy - accuracy) * 100:.2f}%")

# Class-wise improvement
print("\nClass-wise Improvement:")
for i in range(10):
    orig_acc = class_correct[i] / class_total[i] * 100
    imp_acc = improved_class_correct[i] / improved_class_total[i] * 100
    print(f"{classes[i]}: {imp_acc - orig_acc:.2f}%")
```

### Requirements

1. Your submission should include a Jupyter notebook with your code, results, and analysis
2. Make sure to include visualizations of sample images and model performance
3. Document any additional techniques you used to improve model performance
4. Analyze the strengths and weaknesses of your model

## Submission

- Submit the URL of the GitHub Repository that contains your work to NTU black board.
- Should you reference the work of your classmate(s) or online resources, give them credit by adding either the name of your classmate or URL.
