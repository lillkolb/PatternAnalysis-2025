import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
import pickle
import os
from tqdm import tqdm
from functools import partial
import argparse

from dataset import ADNIDatasetTrain, ADNIDatasetTest
from modules import GFNet
from utils import get_transforms

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--trainpath", default="./drive/MyDrive/Colab_Notebooks/AD_NC/train", help="Filepath to ADNI training dataset")
parser.add_argument("-sp", "--savepath", default="./drive/MyDrive/Colab_Notebooks/Final_proj_stored", help="Filepath to saved elements")
parser.add_argument("-s", "--seed", default=10, type=int, help="Seed for reproducibility")
args = parser.parse_args()

# Constants ==============
MAX_EPOCHS = 75
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EARLY_STOP_VAL = 12

# Params ==========
disable_tqdm = False
test_seed = args.seed
train_path = args.trainpath
saving_filepath = args.savepath

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed: int):
    """
    Sets the seed of the random elements of the code in order to maintain consistency between trainings
    """
    torch.manual_seed(seed)                     # sets the seed for RNG on device
    torch.cuda.manual_seed(seed)                # sets the seed for current GPU
    torch.cuda.manual_seed_all(seed)            # sets the seed for all GPUs
    np.random.seed(seed)                        # sets the seed for Numpy library
    random.seed(seed)                           # sets the seed for Random library
    torch.backends.cudnn.deterministic = True   # sets cuDNN to only use deterministic convolution
    torch.backends.cudnn.benchmark = False      # sets cuDNN to not perform benchmarking

# https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
# train 1 epoch function
def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device="cuda", visualise=False):
    """
    
    """
    model.train()
    total_train_loss = 0
    pred_correct = 0
    batch_total = 0

    # i -> batch number, images -> img list for batch, labels -> labels list for batch
    for images, labels in tqdm(train_loader, disable=not visualise): # for each batch of images
        # send the batch to the device
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images)             # 1. forward pass
        outputs = outputs.squeeze(1)        # squeeze outputs to match [64] shape of labels
        loss = criterion(outputs, labels)   # 2. loss calculation
        optimizer.zero_grad()               # 3. zero the parameter gradients before backwards pass
        loss.backward()                     # 4. backwards pass
        optimizer.step()                    # 5. optimiser

        # update params
        total_train_loss += loss.item()     # add current loss to total loss
        predicted = (outputs >= 0).float()  # converting predictions to floats (1.0 or 0.0)
        batch_total += labels.size(0)       # add current batch size
        pred_correct += (predicted == labels).sum().item()
                                            # add the total correct predictions in current batch                                            
    scheduler.step()                        # 6. scheduler

    avg_loss = total_train_loss / len(train_loader)     # average loss throughout epoch
    train_accuracy = pred_correct / batch_total               # model accuracy of epoch

    return avg_loss, train_accuracy

# evaluate model using validation set
def validate_model(model, valid_loader, criterion, device="cuda", visualise=False):
    model.eval()
    total_valid_loss = 0
    pred_correct = 0
    batch_total = 0
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, disable=not visualise):
            # send the batch to the device
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)             # 1. forward pass
            outputs = outputs.squeeze(1)        # squeeze outputs to match [64] shape of labels
            loss = criterion(outputs, labels)   # 2. loss calculation

            # update params
            total_valid_loss += loss.item()     # add current loss to total loss
            predicted = (outputs >= 0).float()  # converting predictions to floats (1.0 or 0.0)
            batch_total += labels.size(0)       # add current batch size
            pred_correct += (predicted == labels).sum().item()
                                            # add the total correct predictions in current batch

    avg_loss = total_valid_loss / len(valid_loader)     # average loss
    valid_accuracy = pred_correct / batch_total               # model accuracy

    return avg_loss, valid_accuracy

def main():
    set_seed(test_seed)

    train_transforms = get_transforms(True)
    test_transforms = get_transforms(False)

    train_dataset = ADNIDatasetTrain(train_path, valid = False, transform=train_transforms, tqdm_disable=disable_tqdm)
    valid_dataset = ADNIDatasetTrain(train_path, valid = True, transform=train_transforms, tqdm_disable=disable_tqdm)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=6)

    model = GFNet(
        img_size=210, 
        patch_size=14, 
        in_chans=1, 
        num_classes=1, 
        embed_dim=384, 
        depth=12,
        mlp_ratio=4., 
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=20, eta_min=1e-6)

    train_loss_data = []
    train_accuracy_data = []
    valid_loss_data = []
    valid_accuracy_data = []
    top_valid_acc = 0
    early_stop_count = 0

    for epoch in range(MAX_EPOCHS):
        
        train_loss, train_accuracy = train_one_epoch(epoch, model, train_loader, criterion, optimiser, scheduler, device=device, visualise=True)
        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device=device, visualise=True)

        train_loss_data.append(train_loss)
        train_accuracy_data.append(train_accuracy)
        valid_loss_data.append(valid_loss)
        valid_accuracy_data.append(valid_accuracy)

        print(f"Epoch: {epoch+1} Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        # save the model if it is better than the current model
        if top_valid_acc < valid_accuracy:
            early_stop_count = 0
            top_valid_acc = valid_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(saving_filepath, 'gfnet_model_test.pt'))
        else:
            early_stop_count += 1

        # stop the training early if there is no improvement in the last epochs
        if (early_stop_count > EARLY_STOP_VAL):
            print(f"No improvement in last {EARLY_STOP_VAL} epochs, stopping training")
            break

    # save accuracy and loss data
    with open(os.path.join(saving_filepath, 'train_loss_data.pkl'), 'wb') as f:
        pickle.dump(train_loss_data, f)

    with open(os.path.join(saving_filepath, 'train_accuracy_data.pkl'), 'wb') as f:
        pickle.dump(train_accuracy_data, f)

    with open(os.path.join(saving_filepath, 'valid_loss_data.pkl'), 'wb') as f:
        pickle.dump(valid_loss_data, f)

    with open(os.path.join(saving_filepath, 'valid_accuracy_data.pkl'), 'wb') as f:
        pickle.dump(valid_accuracy_data, f)

if __name__ == "__main__":
    main()