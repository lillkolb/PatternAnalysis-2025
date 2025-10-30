import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from functools import partial
import argparse

from dataset import ADNIDatasetTrain, ADNIDatasetTest
from modules import GFNet
from utils import analyse_data, get_transforms

# argument parser for users to set their own filepaths
parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--testpath", default="./drive/MyDrive/Colab_Notebooks/AD_NC/test", help="Filepath to ADNI testing dataset")
parser.add_argument("-sp", "--savepath", default="./drive/MyDrive/Colab_Notebooks/Final_proj_stored", help="Filepath to saved elements")
parser.add_argument("-mp", "--modelpath", default='./drive/MyDrive/Colab_Notebooks/Final_proj_stored/gfnet_model.pt', help="Filepath to saved model")
args = parser.parse_args()

test_path = args.testpath
saving_filepath = args.savepath
model_path = args.modelpath

def test_model(model, test_dataset, device):
    """
    test the model, returns a list containing actual values and values predicted by model

    :Args:
                epoch (int): current epoch of training
                model (GFNet): model we are training
                train_loader (Dataloader): dataloader containing training data set
                criterion:
                optimiser: 
                scheduler: 
                device (str): device the images and labels are loaded to (CPU/GPU)
                visualise (bool): toggles visualisation of iterative progress
    :Returns:
                tupel(list[int], list[int]): actual_values, predictions_list
    """
    model.eval()

    predictions_list = []
    actual_values = []

    with torch.no_grad():
        for images, labels in tqdm(test_dataset, disable=False, desc="Testing... "):
            # send the batch to the device
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs).mean().item()

            predictions = 1 if outputs >= 0.56 else 0
            predictions_list.append(predictions)
            actual_values.append(labels.item())

    return actual_values, predictions_list

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create test dataset
    test_transforms = get_transforms(False)
    test_dataset = ADNIDatasetTest(test_path, transform=test_transforms, tqdm_disable=False)

    # create GFNet model instance with the same parameters as training set
    model = model = GFNet(
        img_size=210, 
        patch_size=14, 
        in_chans=1, 
        num_classes=1, 
        embed_dim=384, 
        depth=12,
        mlp_ratio=4., 
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).to(device)

    # load saved model weights from training
    model.load_state_dict(torch.load(model_path))

    # Run model on testing data set
    actual_values, predictions_list = test_model(model, test_dataset, device)

    # Run data analysing processes for model evaluation
    analyse_data(actual_values, predictions_list, saving_filepath)

if __name__ == "__main__":
    main()