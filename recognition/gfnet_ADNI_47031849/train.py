"""
From Task Sheet:
"containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training"
"""

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader

MEAN = 0.11486841564676334
STD = 0.21826585544938487

# Params ==========
disable_tqdm = False
test_seed = 0
train_path = './drive/MyDrive/Colab_Notebooks/AD_NC/train'
test_path = './drive/MyDrive/Colab_Notebooks/AD_NC/test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed: int):
    """
    Sets the seed of the random elements of the code in order to maintain consistency between trainings
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Use this for CUDA
    np.random.seed(seed)
    random.seed(seed)

# https://apxml.com/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/data-augmentation-pytorch-torchvision
# apply geometric transforms before colour
def get_transforms(train):
    if train:
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=(210, 210), scale=(0.95, 1.02), ratio=(0.95, 1.05)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.98, 1.02)),
            transforms.RandomApply([transforms.ElasticTransform(alpha=10.0, sigma=3.0)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.4, scale=(0.01, 0.10), ratio=(0.5, 2.0)), # has to be after tensor
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])
    return data_transforms

set_seed(test_seed)

train_transforms = get_transforms(True)
test_transforms = get_transforms(False)

train_dataset = ADNIDatasetTrain(train_path, valid = False, transform=train_transforms, tqdm_disable=disable_tqdm)
valid_dataset = ADNIDatasetTrain(train_path, valid = True, transform=train_transforms, tqdm_disable=disable_tqdm)
test_dataset = ADNIDatasetTest(test_path, transform=test_transforms, tqdm_disable=disable_tqdm)
