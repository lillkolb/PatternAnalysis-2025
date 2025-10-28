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
import torch.optim as optim
import copy
import pickle
import os
import tqdm

MEAN = 0.11486841564676334
STD = 0.21826585544938487

MAX_EPOCHS = 75
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4

# Params ==========
disable_tqdm = False
test_seed = 10
train_path = './drive/MyDrive/Colab_Notebooks/AD_NC/train'
test_path = './drive/MyDrive/Colab_Notebooks/AD_NC/test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def set_seed(seed: int):
    """
    Sets the seed of the random elements of the code in order to maintain consistency between trainings
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Use this for CUDA
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

        # optimizer.zero_grad()

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

set_seed(test_seed)

train_transforms = get_transforms(True)
test_transforms = get_transforms(False)

train_dataset = ADNIDatasetTrain(train_path, valid = False, transform=train_transforms, tqdm_disable=disable_tqdm)
valid_dataset = ADNIDatasetTrain(train_path, valid = True, transform=train_transforms, tqdm_disable=disable_tqdm)
test_dataset = ADNIDatasetTest(test_path, transform=test_transforms, tqdm_disable=disable_tqdm)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=6)

criterion = nn.BCEWithLogitsLoss()
optimiser = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=20, eta_min=1e-6)

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

saving_filepath = './drive/MyDrive/Colab_Notebooks/Final_proj_stored'
train_loss_data = []
train_accuracy_data = []
valid_loss_data = []
valid_accuracy_data = []
top_valid_acc = 0
early_stop_count = 0
EARLY_STOP_VAL = 12

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
        torch.save(best_model_wts, os.path.join(saving_filepath, 'gfnet_model.pt'))
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

break