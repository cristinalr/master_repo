# Importing necessary libraries and modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torch.nn import Linear, Conv2d, Sequential, Flatten, MaxPool2d, ReLU, AdaptiveAvgPool2d, BatchNorm1d, Dropout1d
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix, accuracy_score

from dataloader_classes import *
from get_dataloaders import *
from get_dataloaders_cv import *



def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('path_folds', type=str, help='Path to csv with folds')
    parser.add_argument('path_test', type=str, help='Path to csv with test data')
    parser.add_argument('path_save_cv_results', type=str, help='Path to csave results from cross validation')
    args = parser.parse_args()
    return args


def train_and_evaluate_fold(dataloaders, criterion, optimizer_conv, scheduler, num_epochs=25):
    # Initialize the model for this fold
    weights = ResNet50_Weights.IMAGENET1K_V2
    model_conv = resnet50(weights=weights)

    # Loads a pretrained ResNet50 model and freezes its parameters to avoid updating them during training
    for param in model_conv.parameters():
        param.requires_grad = False

    # Replaces the final layer of the model to match the number of classes in the dataset
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = Sequential(
        BatchNorm1d(num_ftrs),
        Dropout1d(p=0.3), # Dropout rate might need tuning
        Linear(num_ftrs, 2)  # Assuming binary classification
    )

    # Sets the device to GPU if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_conv = model_conv.to(device)

    # Configure the optimizer to only update parameters of the final layer
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Set up a learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    
    best_acc = 0.0
    metrics = {'loss_train': 0, 'loss_val': 0, 'acc_train': 0, 'acc_val': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model_conv.train()  # Set model to training mode
            else:
                model_conv.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_conv.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_conv(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_conv.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.view(-1).tolist())
                all_labels.extend(labels.view(-1).tolist())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                conf_matrix = confusion_matrix(all_labels, all_preds)
                tn, fp, fn, tp = conf_matrix.ravel()
                metrics.update({'loss_val': epoch_loss, 'acc_val': float(epoch_acc), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})
            else:
                metrics.update({'loss_train': epoch_loss, 'acc_train': float(epoch_acc)})
            # Track the best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

    print('Finished Training Fold')
    return metrics['loss_train'], metrics['loss_val'], metrics['acc_train'], metrics['acc_val'], metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn'], model_conv

def testing_data(model, test_dataloader, criterion):

    model.eval()  
    device = next(model.parameters()).device  # Get the device model is on

    running_corrects = 0
    all_preds = []
    all_labels = []
    running_loss = 0.0

    # Iterate over the test data
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        all_preds.extend(preds.cpu().view(-1).tolist())
        all_labels.extend(labels.cpu().view(-1).tolist())

    # Calculate total accuracy and confusion matrix
    total_samples = len(test_dataloader.dataset)
    accuracy = running_corrects.double() / total_samples
    conf_matrix = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = conf_matrix.ravel()

    results = {
        'accuracy': accuracy.item(),  # Converts to Python float
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return float(accuracy), tp, tn, fp, fn



if __name__ == '__main__':
    # Configures CUDA to optimize computations if available
    cudnn.benchmark = True
    # Enables interactive plotting mode in matplotlib
    plt.ion()

    # Sets the device to GPU if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()

    # Initialize paths from arguments
    path_save = args.path_save_cv_results
    path_folds = args.path_folds
    path_test = args.path_test

    weights = ResNet50_Weights.IMAGENET1K_V2
    model_conv = resnet50(weights=weights)

    # Modify the final layer of the model
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = Sequential(
        BatchNorm1d(num_ftrs),
        Dropout1d(p=0.3),  # Check this value
        Linear(num_ftrs, 2),
    )

    # Moves the model to the configured device (GPU/CPU)
    model_conv = model_conv.to(device)

    # Defines the loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Configures the optimizer and scheduler
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # Load fold data
    folds_df = pd.read_csv(path_folds)
    num_folds = 5

    # Initialize lists to store results per fold
    fold_losses_train = []
    fold_losses_val = []
    fold_accuracies_train = []
    fold_accuracies_val = []
    fold_accuracies = []
    fold_true_positives = []
    fold_true_negatives = []
    fold_false_positives = []
    fold_false_negatives = []

    for fold in range(1, num_folds + 1):
        print(f"Processing fold {fold}")
        cv_dataloaders = get_dataloader_cv(path_test, folds_df, fold, batch_size_train=4, batch_size_val=1, shuffle=True)
        loss_train, loss_val, acc_train, acc_val, tp, tn, fp, fn, model_thisfold = train_and_evaluate_fold(cv_dataloaders, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

        # Testing phase
        accuracy_test, tp_test, tn_test, fp_test, fn_test = testing_data(model_thisfold, cv_dataloaders['test'], criterion)
        print(f"Test Data Fold {fold} - Accuracy: {accuracy_test}, TP: {tp_test}, TN: {tn_test}, FP: {fp_test}, FN: {fn_test}")

        # Append results from training, validation, and test data
        fold_losses_train.append(loss_train)
        fold_losses_val.append(loss_val)
        fold_accuracies_train.append(acc_train)
        fold_accuracies_val.append(acc_val)
        fold_accuracies.append(accuracy_test)
        fold_true_positives.append(tp_test)
        fold_true_negatives.append(tn_test)
        fold_false_positives.append(fp_test)
        fold_false_negatives.append(fn_test)

    # Dictionary to store the results
    results = {
        'Fold': range(1, num_folds + 1),
        'Loss_Train': fold_losses_train,
        'Loss_Val': fold_losses_val,
        'Accuracy_Train': fold_accuracies_train,
        'Accuracy_Val': fold_accuracies_val,
        'Accuracy_Test': fold_accuracies,
        'True_Positives_Test': fold_true_positives,
        'True_Negatives_Test': fold_true_negatives,
        'False_Positives_Test': fold_false_positives,
        'False_Negatives_Test': fold_false_negatives,
    }

    # Create a DataFrame from the dictionary and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(path_save, index=False)
    print(f"Results saved to {path_save}")