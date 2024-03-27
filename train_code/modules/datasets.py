from torch.utils.data import Dataset
from modules.utils import load_json
import numpy as np
from PIL import Image
import os
import pandas as pd
import torchvision.transforms as transforms
from sklearn.model_selection import GroupShuffleSplit
import cv2
import albumentations as A
import torch

def csv_preprocessing(df, image_dir):
    df["image_path"] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
    df = pd.get_dummies(df, columns=['operator'], prefix="operator")
    df["sex"].replace({"M": 0, "F": 1}, inplace=True)
    df["bmi"].fillna(20.05, inplace=True)
    df.fillna(0, inplace=True)
    return df

def train_val_split_by_patient(df, nfold=1, val_ratio=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=nfold, test_size=val_ratio, random_state=random_state)
    for i, (_, test_index) in enumerate(gss.split(X=df, y=None, groups=df['ID'])):
        df.loc[test_index, 'fold'] = i + 1
    df['fold'].fillna(0, inplace=True)
    df['fold'] = df['fold'].astype('int')
    return df

class CustomDataset(Dataset):
    def __init__(self, df, transform=None, mode='train'):
        self.df = df
        self.mode = mode
        self.transform = transform
        self.meta_cols = [col for col in df.columns if col not in ['filename', 'date', 'ID', 'time_min', 'image_path', 'fold']]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = row["image_path"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ###################################################
        if  row["ext_tooth"] == 38:
            T = A.HorizontalFlip(always_apply=True)
            transformed = T(image=image)
            image = transformed["image"]
        ###################################################
        
        
        image = self.transform(image=image)["image"]
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)

        meta_info = row[self.meta_cols].tolist()

        meta_info = torch.tensor(meta_info, dtype=torch.float32)

        if self.mode in ['train', 'val']:
            target = torch.tensor(row["time_min"], dtype=torch.float32)
            return image, meta_info, target, os.path.basename(image_path)

        else:
            return image, meta_info, os.path.basename(image_path)
    
if __name__ == '__main__':
    pass

        