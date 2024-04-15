import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch import nn
from collections import defaultdict
import torchvision.transforms as transforms
import torch
from IPython.display import display
from baukit import TraceDict

class BrainTumorCLF(nn.Module):
    """ Paper's Model """
    def __init__(self, num_classes):
        super(BrainTumorCLF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(6272, 64)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(64, num_classes)
    
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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(64 * 8 * 8, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128, num_classes) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.relu3(x) 
        x = x.view(x.size(0), -1)
        x = self.linear2(self.relu4(self.linear1(x)))
        return x

def plot_histograms(title, datalist):
    fig, axes = plt.subplots(len(datalist), 1, figsize=(10, 1.5 * len(datalist)), sharex=True)
    fig.suptitle(title)
    for i, (name, data) in enumerate(datalist):
        axes[i].hist(data.flatten().detach().numpy(), bins=100)
        axes[i].set_title(name)
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, loss_fn, optimizer, plot_grads=False):
    model.train()
    # initiate a loss monitor
    train_loss = []
    correct_predictions = 0
    
    for images, labels in train_loader:
        # predict the class
        predicted = model(images)
        loss = loss_fn(predicted, labels)
        correct_predictions += (predicted.argmax(dim=1) == labels).sum().item()

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # plot gradients
    if plot_grads:
        with TraceDict(model, [n for n, _ in model.named_modules() if 'conv' in n or 'lin' in n],
                    retain_grad=True) as trace:
            predicted = model(images[0:1])
            loss = loss_fn(predicted, labels[0:1])
            model.zero_grad()
            loss.backward()

        plot_histograms('Parameter gradients', [(n, p.grad)
            for n, p in model.named_parameters() if 'weight' in n])

        plot_histograms('Activations', [(n, trace[n].output)
            for n, p in model.named_modules() if 'lin' in n or 'conv' in n])

        plot_histograms('Activation gradients', [(n, trace[n].output.grad)
            for n, p in model.named_modules() if 'lin' in n or 'conv' in n])
    
    return np.mean(train_loss), correct_predictions / len(train_loader.dataset)

# # training the model
# def train_model(model, train_loader, loss_fn, optimizer):
#     model.train()
#     # Initiate a loss monitor
#     train_loss = []
#     correct_predictions = 0
#     for images, labels in train_loader:
#         # predict the class
#         predicted = model(images)
#         loss = loss_fn(predicted, labels)
#         correct_predictions += (predicted.argmax(dim=1) == labels).sum().item()

#         # Backward pass (back propagation)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         train_loss.append(loss.item())

#     return np.mean(train_loss), correct_predictions / len(train_loader.dataset)

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


def visualize_con_layers_adjusted(model, device, test_data_loader):
    # get batch of images and labels
    images, labels = next(iter(test_data_loader))
    
    def get_activation(layer_name):
        def hook(model, input, output):
            activations[layer_name] = output.detach()
        return hook

    activations = {}
    model.conv1.register_forward_hook(get_activation('first_conv_layer'))
    model.conv2.register_forward_hook(get_activation('second_conv_layer'))
    model.conv3.register_forward_hook(get_activation('third_conv_layer'))

    # run a forward pass
    model.eval()
    with torch.no_grad():
        # Modify this line to ensure images are loaded to the appropriate device
        _ = model(images.to(device))  # Use .to(device) instead of explicitly .cuda()

    # extract the activation of the first image in the batch from the first conv layer
    first_image_features = activations['first_conv_layer'][0]
    second_image_features = activations['second_conv_layer'][0]
    third_image_features = activations['third_conv_layer'][0]
    images_features_lst = [first_image_features, second_image_features, third_image_features]

    # number of feature maps to display
    num_feature_maps_lst = [first_image_features.shape[0], second_image_features.shape[0], third_image_features.shape[0]]

    nth = {
        1: "First",
        2: "Second",
        3: "Third",
        # etc
    }

    for layer in range(len(num_feature_maps_lst)):

        fig, axes = plt.subplots(nrows=int(num_feature_maps_lst[layer]**0.5), ncols=int(num_feature_maps_lst[layer]**0.5), figsize=(12, 12))
        fig.suptitle(f'The {nth[layer+1]} Convolution Layer', fontsize=12)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Plot the feature map
            ax.imshow(images_features_lst[layer][i].cpu().numpy(), cmap='gray')
            ax.axis('off')  # Turn off axis
            if i >= num_feature_maps_lst[layer] - 1:
                break
        plt.show()

def visualize_con_layers_paper(model, device, test_data_loader):
    # get batch of images and labels
    images, labels = next(iter(test_data_loader))
    
    def get_activation(layer_name):
        def hook(model, input, output):
            activations[layer_name] = output.detach()
        return hook

    activations = {}
    model.conv1.register_forward_hook(get_activation('first_conv_layer'))
    model.conv2.register_forward_hook(get_activation('second_conv_layer'))

    # run a forward pass
    model.eval()
    with torch.no_grad():
        # Modify this line to ensure images are loaded to the appropriate device
        _ = model(images.to(device))  # Use .to(device) instead of explicitly .cuda()

    # extract the activation of the first image in the batch from the first conv layer
    first_image_features = activations['first_conv_layer'][0]
    second_image_features = activations['second_conv_layer'][0]
    images_features_lst = [first_image_features, second_image_features]

    # number of feature maps to display
    num_feature_maps_lst = [first_image_features.shape[0], second_image_features.shape[0]]

    nth = {
        1: "First",
        2: "Second",
    }

    for layer in range(len(num_feature_maps_lst)):

        fig, axes = plt.subplots(nrows=int(num_feature_maps_lst[layer]**0.5), ncols=int(num_feature_maps_lst[layer]**0.5), figsize=(12, 12))
        fig.suptitle(f'The {nth[layer+1]} Convolution Layer', fontsize=12)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Plot the feature map
            ax.imshow(images_features_lst[layer][i].cpu().numpy(), cmap='gray')
            ax.axis('off')  # Turn off axis
            if i >= num_feature_maps_lst[layer] - 1:
                break
        plt.show()

def plot_train_val_graphs(epochs, training_losses, training_accuracy, validation_losses, validation_accuracy):# Plotting training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), training_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), training_accuracy, label='Training Accuracy')
    plt.plot(range(1, epochs+1), validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()
