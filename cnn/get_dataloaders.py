import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import sys
import matplotlib.pyplot as plt
from dataloader_classes import *
import torchvision


def get_dataloaders(path_train, path_test, path_val, batch_sizes={'train': 4, 'test': 1, 'val': 1}, shuffle=True):
    # Define transformations
    training_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch Tensor
        torchvision.transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Loading CSV files
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)
    val_df = pd.read_csv(path_val)
    # csv_file = pd.read_csv(csv_file)

    print(f'Training size: {train_df.shape}')

    # Filtering CSV based on IDs to separate train, test, and val
    train_ids = train_df["ID"]
    # train_csv = csv_file[~csv_file["ID"].isin(train_ids)]
    # print(f'Train_csv: {train_csv.shape}')
    
    test_ids = test_df["ID"]
    # test_csv = csv_file[~csv_file["ID"].isin(test_ids)]
    
    val_ids = val_df["ID"]
    # val_csv = csv_file[~csv_file["ID"].isin(val_ids)]

    # Initialize datasets with transformations
    train_dataset = ImageTrainingDataset(csv_file=train_df, transform=training_transform)
    test_dataset = ImageTestingDataset(csv_file=test_df, transform=test_transform)
    val_dataset = ImageTrainingDataset(csv_file=val_df, transform=test_transform)  

    # Initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes['train'], shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes['test'], shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_sizes['val'], shuffle=shuffle)

    dataloaders = {'train': train_dataloader, 'test': test_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset), 'val': len(val_dataset)}

    return dataloaders, dataset_sizes

# def get_dataloaders_cv(train_df, test_df, val_df, batch_sizes={'train': 4, 'test': 1, 'val': 1}, shuffle=True):
#     # Define transformations
#     training_transform = transforms.Compose([
#         transforms.ToTensor(),  # Convert image to PyTorch Tensor
#         torchvision.transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), interpolation=transforms.InterpolationMode.BILINEAR),
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     print(f'Training size: {train_df.shape}')

#     # Filtering CSV based on IDs to separate train, test, and val
#     train_ids = train_df["ID"]
#     # train_csv = csv_file[~csv_file["ID"].isin(train_ids)]
#     # print(f'Train_csv: {train_csv.shape}')
    
#     test_ids = test_df["ID"]
#     # test_csv = csv_file[~csv_file["ID"].isin(test_ids)]
    
#     val_ids = val_df["ID"]
#     # val_csv = csv_file[~csv_file["ID"].isin(val_ids)]

#     # Initialize datasets with transformations
#     train_dataset = ImageTrainingDataset(csv_file=train_df, transform=training_transform)
#     test_dataset = ImageTestingDataset(csv_file=test_df, transform=test_transform)
#     val_dataset = ImageTrainingDataset(csv_file=val_df, transform=test_transform)  

#     # Initialize dataloaders
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes['train'], shuffle=shuffle)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_sizes['test'], shuffle=shuffle)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_sizes['val'], shuffle=shuffle)

#     dataloaders = {'train': train_dataloader, 'test': test_dataloader, 'val': val_dataloader}
#     dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset), 'val': len(val_dataset)}

#     return dataloaders, dataset_sizes