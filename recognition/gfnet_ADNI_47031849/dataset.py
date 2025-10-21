"""
From Task Sheet:
"Containing the data loader for loading and preprocessing your data"
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm # terminal progress bar this library lets us visualise the progress of an iterable
import cv2
import os
import numpy as np # Import numpy for image processing

class ADNI_Dataset(Dataset):
    """

    """

    def __init__(self, img_dir, split="train", transform=None, value=False, seed=1, splitratio=0.75, tqdm_disable=False): # change ratio if needed
        self.img_dir = img_dir
        self.ad = img_dir + '/AD'
        self.nc = img_dir + '/NC'
        self.processed_ad = img_dir + '/processed_AD'
        self.processed_nc = img_dir + '/processed_NC'

    def _preprocess_data(self):
        # check processed AD directory exists
        if (not os.path.exists(self.processed_ad)):
            print("Preprocessing AD images")
            os.mkdir(self.processed_ad)
            for img in tqdm(os.listdir(self.ad), disable=tqdm_disable):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.ad, img)
                    new_path = os.path.join(self.processed_ad, img)
                    self._process_img(img_path, new_path)

        # check processed AD directory exists
        if (not os.path.exists(self.processed_nc)):
            print("Preprocessing NC images")
            os.mkdir(self.processed_nc)
            for img in tqdm(os.listdir(self.nc), disable=tqdm_disable):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.nc, img)
                    new_path = os.path.join(self.processed_nc, img)
                    self._process_img(img_path, new_path)


    def _process_img(self, raw_filepath, processed_filepath):
        # load in grayscale (since images are grayscale)
        image = cv2.imread(raw_filepath, cv2.IMREAD_GRAYSCALE)

        # Add the cropping logic here
        cropped_image = self._crop_brain_region(image)

        # Save the processed image
        cv2.imwrite(processed_filepath, cropped_image)

    def _crop_brain_region(self, image):
        # threshold the image using Otsu's threshold method
        _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the coordinates of non-zero pixels
        coords = cv2.findNonZero(binary_mask)

        if coords is not None:
            # Get the bounding box of the non-zero pixels
            x, y, w, h = cv2.boundingRect(coords)

            # Crop the image using the bounding box
            cropped_image = image[y:y+h, x:x+w]
            return cropped_image
        else:
            # If no non-zero pixels are found, return the original image or handle as appropriate
            print("Error! Empty image - preprocessing has not been applied")
            return image