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

def get_dataloader_cv(path_test, csv_file, fold_number, batch_size_train=4, batch_size_val=1, shuffle=True):
    
    # Define transformations
    training_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_df = pd.read_csv(path_test)
    test_dataset = ImageTestingDataset(csv_file=test_df, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle)

    # Split data based on the 'fold' column
    train_df = csv_file[csv_file['fold'] != fold_number]
    val_df = csv_file[csv_file['fold'] == fold_number]

    
    # Create datasets
    train_dataset = ImageTrainingDataset(csv_file=train_df, transform=training_transform)
    val_dataset = ImageTrainingDataset(csv_file=val_df, transform=test_transform)  # Using test_transform for consistency
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)  # No shuffle for validation set
    
    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}