import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
from pytorch_toolbelt import losses as L
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import cv2
from model import Model
import os
import gc
import glob
import os.path as osp

def rle_decode(mask_rle, shape=(768, 768), resize=False):

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    img = img.reshape(shape).T
    if resize:
        img = cv2.resize(img, (224, 224), cv2.INTER_AREA)
    return img

# Function for decoding a csv file with masks into masks images
def mask_as_image(masks, resize=True):
    possible_masks = np.zeros((768, 768))
    for mask in masks:
        if isinstance(mask, float):
            break
        else:
            mask_image = rle_decode(mask, resize=True)
            if resize:
                mask_image = cv2.resize(mask_image, (768,768), cv2.INTER_AREA)
                mask_image = cv2.normalize(mask_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

            possible_masks += mask_image
    return possible_masks



# Code for dataset initialization
class shipDataset(Dataset):
    def __init__(self, paths, transforms=None, train=True, test=False, mask_file=None):
        self.paths = paths
        self.transforms = transforms
        self.train = train
        self.mask_file = mask_file
        self.test = test
        self.permute = False

    def __len__(self):
        return len(self.paths)

    # Function for reading and changing images from a dataset
    def __getitem__(self, idx):
        p = self.paths[idx]
        imageId = p.split('/')[-1]
        image = cv2.imread(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image = cv2.resize(image, (768, 768), cv2.INTER_AREA)
        if self.train:
            mask_list = self.mask_file.loc[self.mask_file["ImageId"] == imageId, "EncodedPixels"].tolist()
            masks = mask_as_image(mask_list, resize=True)
            if self.transforms:
                image = self.transforms(image=image)
                masks = self.transforms(masks)

            masks = torch.from_numpy(masks).float()
        image = torch.from_numpy(image).float()

        if self.permute:
            image = image.permute(2, 0, 1)
        if self.test:
            return image, imageId

        else:
            return image, masks

def get_train(paths, next_path, csv, ship=True):
    csv_mask = csv.copy()
    new_lst = []
    if ship:
        df = csv_mask[csv_mask["EncodedPixels"].notna()].copy()
    else:
        df = csv_mask[csv_mask["EncodedPixels"].isna()].copy()
    df = df.drop_duplicates(subset="ImageId")
    for rows in (df.iterrows()):
        new_lst.append(next_path + "/" + rows[1].ImageId)
    return new_lst

filenames_train = glob.glob(osp.join('train_v2', '*.jpg'))
test_paths = glob.glob(osp.join('test_v2', '*.jpg'))
mask_file = pd.read_csv('train_ship_segmentations_v2.csv')


# Reduce train size
train_ship = get_train(filenames_train, "train_v2", mask_file, True)
train_na_ship = get_train(filenames_train, "train_v2", mask_file, False)

# Reduce size of train with and without ships
train_na_ship = random.sample(train_na_ship, int(len(train_na_ship) * 0.01))
train_ship = random.sample(train_ship, int(len(train_ship) * 0.05))
filenames_train = train_na_ship + train_ship

# Train/Val split
train_paths, validation_paths = train_test_split(filenames_train, test_size=0.3, random_state=42)

# DataSets
train_dataset = shipDataset(train_paths, transforms=False, train=True, mask_file=mask_file)
valid_dataset = shipDataset(validation_paths, train=True, mask_file=mask_file)
test_dataset = shipDataset(test_paths, train=False, test=True)

# DataLoaders
BS = 1
NW = 0
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def show_results(model, loader, num_of_rows=16):
    fig, ax = plt.subplots(num_of_rows, 4, figsize=(16, 16))
    row = 0
    model = model.cpu()
    for image, mask in loader:
        image = image.cpu().permute(0, 3, 1, 2)
        pred = model(image)
        m = nn.Softplus()
        pred = m(pred)
        image = image.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
        mask = mask.cpu().detach()[0].numpy()
        pred = pred.detach()[0][0].cpu().numpy()
        ax[row][0].imshow(image)
        ax[row][1].imshow(mask)
        ax[row][2].imshow(pred)
        ax[row][3].imshow(image)
        ax[row][3].imshow(pred, alpha=0.5)
        for j in range(4):
            ax[row][j].axis('off')
        ax[row][0].set_title('image')
        ax[row][1].set_title('mask')
        ax[row][2].set_title('prediction')
        ax[row][3].set_title('image with prediction')
        row += 1
        if row == num_of_rows:
            break
    plt.savefig('evalimg.png')


path = "MyModel.pth"
model = torch.load(path)

show_results(model, train_loader, num_of_rows=8)

