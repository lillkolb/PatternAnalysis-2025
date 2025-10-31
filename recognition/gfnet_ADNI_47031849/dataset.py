import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm # terminal progress bar this library lets us visualise the progress of an iterable
import cv2
import os
import numpy as np # Import numpy for image processing
import random
from collections import defaultdict # essentially a dict that does not raise a KeyError when accessing a nonexistent key

class ADNIDatasetTrain(Dataset):
    """
    ADNI Dataset for training data
    """
    def __init__(self, img_dir, transform=None, valid=False, seed=1, split_ratio=0.75, tqdm_disable=False): # change ratio if needed
        """
        Initialise dataset

        :Args:
                img_dir (str): Path to training directory containing images
                transform (callable, optional): Optional transform to be applied on a sample.
                valid (bool): True if validation set, False if training set
                seed (int): Seed for consistentency in random number generators
                split_ratio (float): how much of the training dataset we split to train on
                tqdm_disable (bool): Whether to disable the tqdm progress bar
        """
        self.img_dir = img_dir
        self.ad = img_dir + '/AD'
        self.nc = img_dir + '/NC'
        self.processed_ad = img_dir + '/processed_AD'
        self.processed_nc = img_dir + '/processed_NC'
        self.tqdm_disable = tqdm_disable

        # run preprocess on raw images
        self._preprocess_data()
        self.ad_images = [os.path.join(self.processed_ad, img) for img in os.listdir(self.processed_ad)]
        self.nc_images = [os.path.join(self.processed_nc, img) for img in os.listdir(self.processed_nc)]
        self.all_images = self.ad_images + self.nc_images
        self.all_labels = [1] * len(self.ad_images) + [0] * len(self.nc_images)

        self.transform = transform
        self.valid = valid
        self.seed = seed
        self.split_ratio = split_ratio

        self.mask = self._create_mask()

    def _create_mask(self):
        """
        Creates a mask for the dataset. Splits the training dataset into training and validation sets.

        :Args: None

        :Returns: List[bool]

        :Raises: None
        """
        random.seed(self.seed)
        if self.split_ratio == 1:
            return [True] * len(self.all_images)
        else:
            # split the data
            train_split = [random.random() < self.split_ratio for _ in range(len(self.all_images))]
            # validation data is images not used in training
            if self.valid:
                return [not x for x in train_split]
            return train_split

    def _preprocess_data(self):
        """
        Preprocess images in AD and NC directories
        These will be stored in processed_AD and processed_NC respectively

        :Args: None

        :Returns: None

        :Raises: None
        """
        # check processed AD directory exists
        if (not os.path.exists(self.processed_ad)):
            print("Preprocessing AD images")
            os.mkdir(self.processed_ad)
            for img in tqdm(os.listdir(self.ad), disable=self.tqdm_disable):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.ad, img)
                    new_path = os.path.join(self.processed_ad, img)
                    self._process_img(img_path, new_path)

        # check processed NC directory exists
        if (not os.path.exists(self.processed_nc)):
            print("Preprocessing NC images")
            os.mkdir(self.processed_nc)
            for img in tqdm(os.listdir(self.nc), disable=self.tqdm_disable):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.nc, img)
                    new_path = os.path.join(self.processed_nc, img)
                    self._process_img(img_path, new_path)

    def _process_img(self, raw_filepath, processed_filepath):
        """
        Process a single image. Image is cropped, resized, padded and saved.

        :Args:
                raw_filepath (str): Path to the raw image file
                processed_filepath (str): Path to save the processed image

        :Returns: None

        :Raises: None
        """
        # load in grayscale (since images are grayscale)
        image = cv2.imread(raw_filepath, cv2.IMREAD_GRAYSCALE)

        # Add the cropping logic here
        cropped_image = self._crop_brain_region(image)

        # we need to make sure all of the images are of the same pixel size.
        height, width = cropped_image.shape

        # resize the largest dimension 210 (since none of the images exceed this in terms of valid region)
        scaling = 210 / max(height, width)
        new_height = int(height * scaling)
        new_width = int(width * scaling)

        # resize image with lancoz interpolation
        cropped_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Find padding values for 210x210 image
        pad_top = (210 - new_height) // 2
        pad_bottom = 210 - new_height - pad_top
        pad_left = (210 - new_width) // 2
        pad_right = 210 - new_width - pad_left

        # Add padding to the image
        pad_img = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        # Save the processed image
        cv2.imwrite(processed_filepath, pad_img)

    def _crop_brain_region(self, image):
        """
        Seperate the brain region from the background and crop the image based
        on the bounding box of the brain region.

        :Args:
                image (np.array): image to be cropped

        :Returns:
                np.array: cropped image

        :Raises: None
        """

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
            print("Error! Empty image - cropping has not been applied")
            return image

    def __len__(self):
        """
        Get length of dataset

        :Args: None

        :Returns:
                int: dataset length

        :Raises: None
        """
        return sum(self.mask)

    def __getitem__(self, idx):
        """
        Get image from dataset

        :Args:
                idx (int): index of image in dataset

        :Returns:
                tuple(np.array, int): image, label

        :Raises: IndexError
        """
        # Find the actual index in the full list based on the mask
        masked_idx = -1
        count = 0
        for i, include in enumerate(self.mask):
            if include:
                if count == idx:
                    masked_idx = i
                    break
                count += 1

        if masked_idx == -1:
            raise IndexError("Index out of bounds for masked dataset")

        img_path = self.all_images[masked_idx]
        label = self.all_labels[masked_idx]

        image = Image.open(img_path).convert('L') # Ensure image is in grayscale format

        if self.transform:
            image = self.transform(image)

        return image, label

class ADNIDatasetTest(Dataset):
    """
    ADNI Dataset for testing data
    """
    def __init__(self, img_dir, transform=None, tqdm_disable=False):
        """
        Initialise dataset

        :Args:
                img_dir (str): Path to training directory containing images
                transform (callable, optional): Optional transform to be applied on a sample.
                tqdm_disable (bool): Whether to disable the tqdm progress bar

        """
        self.img_dir = img_dir
        self.ad = img_dir + '/AD'
        self.nc = img_dir + '/NC'
        self.processed_ad = img_dir + '/processed_AD'
        self.processed_nc = img_dir + '/processed_NC'
        self.tqdm_disable = tqdm_disable
        self.transform = transform

        self._preprocess_data()

        self.img_groups = []
        self.img_groups.extend(self._group_images(self.processed_ad, 1))
        self.img_groups.extend(self._group_images(self.processed_nc, 0))

    def _preprocess_data(self):
        """
        Preprocess images in AD and NC directories
        These will be stored in processed_AD and processed_NC respectively

        :Args: None

        :Returns: None

        :Raises: None
        """
        # check processed AD directory exists
        if (not os.path.exists(self.processed_ad)):
            print("Preprocessing AD images")
            os.mkdir(self.processed_ad)
            for img in tqdm(os.listdir(self.ad), disable=self.tqdm_disable):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.ad, img)
                    new_path = os.path.join(self.processed_ad, img)
                    self._process_img(img_path, new_path)

        # check processed NC directory exists
        if (not os.path.exists(self.processed_nc)):
            print("Preprocessing NC images")
            os.mkdir(self.processed_nc)
            for img in tqdm(os.listdir(self.nc), disable=self.tqdm_disable):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(self.nc, img)
                    new_path = os.path.join(self.processed_nc, img)
                    self._process_img(img_path, new_path)

    def _process_img(self, raw_filepath, processed_filepath):
        """
        Process a single image. Image is cropped, resized, padded and saved.

        :Args:
                raw_filepath (str): Path to the raw image file
                processed_filepath (str): Path to save the processed image

        :Returns: None

        :Raises: None
        """
        # load in grayscale (since images are grayscale)
        image = cv2.imread(raw_filepath, cv2.IMREAD_GRAYSCALE)

        # Add the cropping logic here
        cropped_image = self._crop_brain_region(image)

        # we need to make sure all of the images are of the same pixel size.
        height, width = cropped_image.shape

        # resize the largest dimension 210 (since none of the images exceed this in terms of valid region)
        scaling = 210 / max(height, width)
        new_height = int(height * scaling)
        new_width = int(width * scaling)

        # resize image with lancoz interpolation
        cropped_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Find padding values for 210x210 image
        pad_top = (210 - new_height) // 2
        pad_bottom = 210 - new_height - pad_top
        pad_left = (210 - new_width) // 2
        pad_right = 210 - new_width - pad_left

        # Add padding to the image
        pad_img = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        # Save the processed image
        cv2.imwrite(processed_filepath, pad_img)

    def _crop_brain_region(self, image):
        """
        Seperate the brain region from the background and crop the image based
        on the bounding box of the brain region.

        :Args:
                image (np.array): image to be cropped

        :Returns:
                np.array: cropped image

        :Raises: None
        """
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
            print("Error! Empty image - cropping has not been applied")
            return image

    def _group_images(self, dir, label):
        """
        Test images are given in groups, where the first number corresponds to
        a given brain. Group the images by their number, there should be 20 images per group

        :Args:
                dir (str): processed directory path
                label (int): 1 for AD, 0 for NC

        :Returns:
                List[dict]: a list of dictionaries containing the group number, filenames and label

        :Raises: None
        """
        img_groups = []
        unsorted_groups = defaultdict(list)

        for filename in os.listdir(dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract the group number from the filename
                group_number = filename.split('_')[0] # keep as str
                unsorted_groups[group_number].append(filename)

        # Sort the groups based on the group number
        for group_number, filenames in unsorted_groups.items():
            # sort filenames based on second number of filename
            sorted_group = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]))

            # check group size is 20
            if len(sorted_group) != 20:
                # raise Exception(f"Group {group_number} does not contain excalty 20 images")
                print(f"Group {group_number} does not contain excalty 20 images")

            img_groups.append({
                "group_number" : group_number,
                "filenames" : sorted_group,
                "label" : label
            })

        return img_groups

    def __len__(self):
        """
        Get length of dataset

        :Args: None

        :Returns:
                int: dataset length

        :Raises: None
        """
        return len(self.img_groups)

    def __getitem__(self, idx):
        """
        Get image stack from dataset

        :Args:
                idx (int): index of image in dataset

        :Returns:
                tuple(torch.tensor, torch.tensor): image stack, label
        """
        # grab image and labels
        group = self.img_groups[idx]
        group_num = group["group_number"]
        filenames = group["filenames"]
        label = group["label"]

        # initialise an empty image stack
        images_stacked = []

        # get target directory based on label
        image_dir = self.processed_ad if label == 1 else self.processed_nc

        for filename in filenames:
            img_path = os.path.join(image_dir, filename)

            # open image and convert to 8 bit grayscale
            image = Image.open(img_path).convert('L')

            # apply transform
            if self.transform:
                image = self.transform(image)

            # add image to image stack
            images_stacked.append(image)

        # use numpy to stack tensors since we have a list
        # images stacked to form (20, 1, 210, 210) shape
        images_stacked = np.stack(images_stacked, axis = 0)
        return torch.tensor(images_stacked, dtype=torch.float32), torch.tensor(label).float()
