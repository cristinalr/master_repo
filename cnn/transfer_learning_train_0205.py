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

from dataloader_classes import *
from get_dataloaders import *

"""
CSV_FILE: /mnt/md0/cristina/data/id_match_v3.csv
PATH_TRAIN: /mnt/md0/cristina/data/train_data.csv
PATH TEST: /mnt/md0/cristina/data/test_data.csv
PATH VAL: /mnt/md0/cristina/data/validation_data.csv

python3 transfer_learning_train.py /mnt/md0/cristina/data/train_data_paths.csv /mnt/md0/cristina/data/test_data_paths.csv /mnt/md0/cristina/data/validation_data_paths.csv
"""


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('path_train', type=str, help='Path to training dataset')
    parser.add_argument('path_test', type=str, help='Path to test dataset')
    parser.add_argument('path_val', type=str, help='Path to val dataset')
    parser.add_argument('path_save', type=str, help='Path to save model')
    args = parser.parse_args()
    return args


# Defines the training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, train_dataloader=None, val_dataloader=None):
    since = time.time()  # Records the start time
    loss_train = np.zeros((num_epochs))
    acc_train = np.zeros((num_epochs))
    loss_val = np.zeros((num_epochs))
    acc_val = np.zeros((num_epochs))

    # Creates a temporary directory to save the best model's parameters
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        # Saves the initial model's state
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0  # Initializes the best accuracy

        # Loops over each epoch
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Alternates between training and validation phases
            for phase, dataloader in [('train', train_dataloader), ('val', val_dataloader)]:
                # Sets model to training mode or evaluation mode
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterates over data in batches
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    print(f'Phase: {phase}, Shape: {inputs.shape}')

                    # Resets gradient information to zero
                    optimizer.zero_grad()

                    # Forward pass: Computes predicted outputs by passing inputs to the model
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass and optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Calculates and accumulates the loss and number of correct predictions
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    print(f'Running loss: {running_loss}')

                # Updates the learning rate through the scheduler
                if phase == 'train':
                    scheduler.step()

                # Computes the epoch's loss and accuracy
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # Save arrays with epoch_loss and epoch_acc
                if phase =='train':
                    loss_train[epoch] = epoch_loss
                    acc_train[epoch] = epoch_acc
                elif phase == 'val':
                    loss_val[epoch] = epoch_loss
                    acc_val[epoch] = epoch_acc

                print(f'---\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Updates the best model if the current model is better
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        # Prints out the training time and best validation accuracy
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # Loads the best model's parameters
        model.load_state_dict(torch.load(best_model_params_path))

    # Save model in save_path dir

    return model, loss_train, loss_val, acc_train, acc_val


if __name__ == '__main__':
    # Configures CUDA to optimize computations if available
    cudnn.benchmark = True
    # Enables interactive plotting mode in matplotlib
    plt.ion()

    # Sets the device to GPU if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()

    # csv_file = args.csv_file
    path_train = args.path_train
    path_test = args.path_test
    path_val = args.path_val
    path_save = args.path_save

    # Initializes dataloaders for training and validation, and their respective dataset sizes
    dataloaders, dataset_sizes = get_dataloaders(path_train, path_test, path_val)

    weights = ResNet50_Weights.IMAGENET1K_V2
    model_conv = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Loads a pretrained ResNet50 model and freezes its parameters to avoid updating them during training
    for param in model_conv.parameters():
        param.requires_grad = False

    # Replaces the final layer of the model to match the number of classes in the dataset
    num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, 2)  # Adjust this
    model_conv.fc = Sequential(
        BatchNorm1d(num_ftrs),
        Dropout1d(p=0.3), # Check this value
        Linear(num_ftrs, 2),
    )

    # Moves the model to the configured device (GPU/CPU)
    model_conv = model_conv.to(device)

    # Defines the loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Configures the optimizer to only update parameters of the final layer
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Sets up a learning rate scheduler to decrease the learning rate by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv, loss_train, loss_val, acc_train, acc_val = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25, train_dataloader=dataloaders['train'], val_dataloader=dataloaders['val'])

    if not os.path.isdir(path_save):
        os.makedirs(path_save)
    

    results_dict = {
        'loss_train': loss_train,
        'acc_train': acc_train,
        'loss_val': loss_val,
        'acc_val': acc_val,
    }

    results_df = pd.DataFrame(results_dict, columns = ['loss_train', 'acc_train', 'loss_val', 'acc_val'])

    results_df.to_csv(path_save + 'results_test_0905.csv')

    torch.save(model_conv.state_dict(), path_save+('trained_model.pt'))
