import numpy as np
import pandas as pd
import random
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
from pytorch_toolbelt import losses as L
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset,DataLoader
import cv2
from model import Model
import os
import gc
import glob
import os.path as osp



masks = pd.read_csv('train_ship_segmentations_v2.csv')


# Decoder for csv file with masks
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

# Function for defining files with and without ships
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

# Identification of all picture titles
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

gc.collect()

NUM_EPOCHS = 3

# Function for defining epochs
def get_train_and_valid_epochs(model, loss, metrics, optimizer):
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    return train_epoch, valid_epoch

# Training function over epochs
def train(model, loss, metrics, optimizer, train_loader, valid_loader, model_name):
    max_score = 0
    train_epoch, valid_epoch = get_train_and_valid_epochs(model, loss, metrics, optimizer)

    df = pd.DataFrame(columns=['train_loss', 'train_iou', 'val_loss', 'val_iou'])
    for i in range(0, NUM_EPOCHS):

        print('\nEpoch: {}'.format(i+1))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        epoch_df = pd.DataFrame({'train_loss': [train_logs['loss']], 'train_iou': [train_logs['iou_score']],
                                 'val_loss': [valid_logs['loss']], 'val_iou': [valid_logs['iou_score']]})

        df = pd.concat([df, epoch_df], ignore_index=True)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, f'./{model_name}.pth')
            print('Model saved!')
    df.to_csv(f'{model_name}.csv', index=False)
    return df



# Utils
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = L.DiceLoss(mode='binary')
metrics = [
    IoU(threshold=0.5).to(DEVICE),
]

train_loader.dataset.permute = True
valid_loader.dataset.permute = True

ENCODER_RESNET='resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['airbus']
ACTIVATION = 'sigmoid'


# Running model function
def run_model(model, model_name, loss):
    model.cuda()
    optimizer= torch.optim.Adam(model.parameters(),lr= 1e-3)
    model_name=model_name

    return train(model, loss, metrics, optimizer, train_loader, valid_loader, model_name)

# Model identification
model = Model(in_channels=3, out_chanels=1).to(DEVICE)


if __name__ == '__main__':
    run_model(model, 'MyModel', loss)

gc.collect()
torch.cuda.empty_cache()