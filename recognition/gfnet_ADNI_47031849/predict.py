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

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--testpath", default="./drive/MyDrive/Colab_Notebooks/AD_NC/test", help="Filepath to ADNI testing dataset")
parser.add_argument("-sp", "--savepath", default="./drive/MyDrive/Colab_Notebooks/Final_proj_stored", help="Filepath to saved elements")
args = parser.parse_args()

test_path = args.testpath
saving_filepath = args.savepath

def test_model(model, test_dataset, device):
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

    test_transforms = get_transforms(False)
    test_dataset = ADNIDatasetTest(test_path, transform=test_transforms, tqdm_disable=False)

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

    model.load_state_dict(torch.load('./drive/MyDrive/Colab_Notebooks/Final_proj_stored/gfnet_model_10_797.pt'))

    actual_values, predictions_list = test_model(model, test_dataset, device)

    analyse_data(actual_values, predictions_list, saving_filepath)

if __name__ == "__main__":
    main()