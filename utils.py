import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch import nn
from collections import defaultdict
import torchvision.transforms as transforms
import torch
from IPython.display import display

class BrainTumorCLF(nn.Module):
    """ Paper's Model """
    def __init__(self):
        super(BrainTumorCLF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(6272, 64)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(64, 3)
    
    def forward(self, x):
        res = self.conv1(x)
        res = self.maxpool1(res)
        res = self.conv2(res)
        res = self.maxpool2(res)
        res = self.flatten(res)
        res = self.lin1(res)
        res = self.relu(res)
        res = self.lin2(res)
        out = nn.functional.softmax(res, dim=1)
        return out

class BrainTumorCNN(nn.Module):
    """ Adjusted Model """
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # Setting the output to match the number of classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
# training the model
def train_model(model, train_loader, loss_fn, optimizer):
    model.train()
    # Initiate a loss monitor
    train_loss = []
    correct_predictions = 0
    for images, labels in train_loader:
        # predict the class
        predicted = model(images)
        loss = loss_fn(predicted, labels)
        correct_predictions += (predicted.argmax(dim=1) == labels).sum().item()

        # Backward pass (back propagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    return np.mean(train_loss), correct_predictions / len(train_loader.dataset)

def evaluate_model(model, val_loader, loss_fn, device, return_confusion_matrix=False):
    model.eval()
    # Initiate a loss monitor
    val_loss = []
    true_labels = []
    predicted_labels = []

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        # predict the class
        predicted = model(images)
        loss = loss_fn(predicted, labels)

        val_loss.append(loss.item())
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.argmax(dim=1).cpu().numpy())

    val_loss = np.mean(val_loss)
    val_acc = accuracy_score(true_labels, predicted_labels)
    val_precision = precision_score(true_labels, predicted_labels, average='micro')
    val_recall = recall_score(true_labels, predicted_labels, average='micro')

    if return_confusion_matrix:
        cm = confusion_matrix(true_labels, predicted_labels)
        return val_loss, val_acc, val_precision, val_recall, cm
    else:
        return val_loss, val_acc, val_precision, val_recall

def draw_confusion_matrix(confusion_matrix, val_set, labels, cmap="Blues"):
    confusion_matrix = np.array(confusion_matrix)
    assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
    assert confusion_matrix.shape[0] == len(labels)

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.imshow(confusion_matrix, cmap=cmap)

    ax.set_xticks(np.arange(len(val_set.classes)), labels)
    ax.set_yticks(np.arange(len(val_set.classes)), labels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")

    for true_label in range(len(val_set.classes)):
        for pred_label in range(len(val_set.classes)):
            ax.text(
                pred_label,
                true_label,
                int(confusion_matrix[true_label, pred_label]),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    plt.show()

def find_image(model, true_class, predicted_class, test_loader):
    """ Prints an image with the given true label and prediction for the given model """
    # store images, true labels, and predicted labels for each class
    class_images = defaultdict(list)

    # get predictions
    for images, labels in test_loader:
        # Perform forward pass
        with torch.no_grad():
            outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        # add each image to class images dict
        for image, true_label, predicted_label in zip(images, labels, predicted):
            class_images[true_label.item()].append((image, true_label.item(), predicted_label.item()))

    # find an image that matches the true class and predicted class
    found_image = False
    for image, true_label, predicted_label in class_images[true_class]:
        if true_label == true_class and predicted_label == predicted_class:
            tensor_to_pil = transforms.ToPILImage()
            pil_image = tensor_to_pil(image)
            display(pil_image.resize((256, 256)))
            print(f"True label: {true_label}")
            print(f"Predicted label: {predicted_label}")
            found_image = True
            break

    if not found_image:
        print("No image found matching the specified true class and predicted class.")
