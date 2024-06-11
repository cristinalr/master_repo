import torch
import os
import glob
import pandas as pd
import re
import SimpleITK as sitk
from generate_rotated_mip import load_coronal_mip
import torchvision
from torchvision import transforms
import skimage as ski
import scipy
import numpy as np
pd.options.mode.chained_assignment = None


class ImageTrainingDataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, csv_file, transform=None, drop_last=True):
            super().__init__()
            self.transform = transform

            # Create a DataFrame with case IDs and labels
            self.df = csv_file

            self.df.loc[self.df['patient_group'] == 'HC', 'patient_group'] = 0
            self.df.loc[self.df['patient_group'] == 'PSC', 'patient_group'] = 1



      def __len__(self):
            return len(self.df)

      def __getitem__(self, idx):

            img_loc = self.df["img_path"].iloc[idx]
            img_loc = os.path.join('/mnt/md0/cristina/data/', img_loc)
            label = self.df["patient_group"].iloc[idx]
        
            # Random angle for MIP
            angle = np.random.choice([-20, -15, -10, -5, 0, 5, 10, 15, 20])

            # Generate MIP using function from generate_rotated_mip.py
            mip = load_coronal_mip(img_loc, rotation_angle=angle)

            # Convert MIP to numpy array
            image = sitk.GetArrayFromImage(mip)

            #   np.percentile - winsorization (95%)
            pct95 = np.percentile(image, 95)

            image = image/pct95
            image[image>1] = 1

            # Histogram equalization - sitk
            ski.exposure.equalize_adapthist(image, kernel_size=None, clip_limit=0.015, nbins=256)
     
            #   np.transpose - dimensions (1x256x256)
            image = np.transpose(image, [1,0,2]) 
            
            # Gaussian noise 
            sd = np.random.choice([0,0.01,0.05])
            noise_array = np.random.normal(0,sd,image.shape)
            image += noise_array

            image = np.squeeze(image)
            

            if self.transform:
                  image = self.transform(image)

            img_3ch = torch.zeros((3, image.shape[1], image.shape[2]))
            img_3ch[0,:,:] = image
            img_3ch[1,:,:] = image
            img_3ch[2,:,:] = image
            

            return img_3ch, label

class ImageTestingDataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, csv_file, transform=None):
            self.transform = transform

            # Create a DataFrame with case IDs and labels
            self.df = csv_file
            # self.df['patient_group'] = self.df['patient_group'].replace(['HC'], 0)
            # self.df['patient_group'] = self.df['patient_group'].replace(['PSC'], 1)

            self.df.loc[self.df['patient_group'] == 'HC', 'patient_group'] = 0
            self.df.loc[self.df['patient_group'] == 'PSC', 'patient_group'] = 1


      def __len__(self):
            return len(self.df)

      def __getitem__(self, idx):

            img_loc = self.df["img_path"].iloc[idx]
            img_loc = os.path.join('/mnt/md0/cristina/data/', img_loc)
            label = self.df["patient_group"].iloc[idx]

            # Generate MIP using function from generate_rotated_mip.py
            mip = load_coronal_mip(img_loc)

            # Convert MIP to numpy array
            image = sitk.GetArrayFromImage(mip)

            #   np.percentile - winsorization (95%)
            pct95 = np.percentile(image, 95)
            image = image/pct95
            image[image>1] = 1

            # Histogram equalization - sitk
            ski.exposure.equalize_adapthist(image, kernel_size=None, clip_limit=0.015, nbins=256)
     
            #   np.transpose - dimensions (1x256x256)
            image = np.transpose(image, [1,0,2]) 
            
      
            image = np.squeeze(image)

            if self.transform:
                  image = self.transform(image)

            img_3ch = torch.zeros((3, image.shape[1], image.shape[2]))
            img_3ch[0,:,:] = image
            img_3ch[1,:,:] = image
            img_3ch[2,:,:] = image

            return img_3ch, label