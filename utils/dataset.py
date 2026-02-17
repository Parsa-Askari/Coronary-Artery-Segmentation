from torch.utils.data import Dataset,DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from tqdm.notebook import tqdm
import cv2
import torch
import random
from torch.utils.data import Sampler
###IE###
###SS###

class TrainUnetDataset(Dataset):
    def __init__(self, transform,data,normalizer,example_phase=False):
        super(TrainUnetDataset, self).__init__()
        self.data = data
        self.transform = transform
        self.normalizer = normalizer
        self.example_phase = example_phase
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        img : (H,W,C)
        maks : (H,W)
        """
        img, mask,_ = self.data[index]
        
        if img.ndim == 2: img = img[..., None]
        if mask.ndim == 2: mask = mask[..., None]

        result = self.transform(image=img, mask=mask)
        new_image_not_normal  = result['image']
        new_mask= result['mask']

        new_image = self.normalizer(image = new_image_not_normal)["image"]

        multi_mask = torch.from_numpy(new_mask).squeeze(-1).long() #(H,W)
        binary_mask = (multi_mask !=0).squeeze(-1).long() #(H,W)
        

        if torch.is_tensor(new_image): new_image = new_image.numpy()
        
        # turn image shape to : (C,H,W)
        if new_image.ndim == 3 : 
            new_image = new_image.transpose(2, 0, 1) 
        elif new_image.ndim == 2:
            new_image = new_image[None, :, :] 

        if(self.example_phase):
            return new_image_not_normal,new_image, binary_mask, multi_mask
        
        return new_image, binary_mask, multi_mask

class UnetExampleDataset(Dataset):
    def __init__(self,transform,data,base_transform=None):
        super(UnetExampleDataset,self).__init__()
        
        self.data = data
        self.transform = transform
        self.to_tensor = ToTensorV2()
        if(base_transform is None):
            self.base_transform = A.Compose([ToTensorV2()])
        else:
            self.base_transform = base_transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img , mask = self.data[index]
        img = np.expand_dims(img, axis=-1) 
        result = self.transform(image=img, mask=mask)
        new_image = result['image']
        new_mask = result['mask']
        
        raw_result = self.base_transform(image=img, mask=mask)
        raw_image = raw_result['image']
        raw_mask = raw_result['mask']

        new_image = self.to_tensor(image = new_image)["image"]
        return new_image.float() , new_mask , raw_image.float() , raw_mask

if __name__ == "__main__":
    train_transforms = A.Compose([
        A.GaussianBlur(
            sigma_limit=[0.1,0.5],
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.15,
            brightness_by_max=True,
            p=0.3
        ),
        A.RandomGamma(
            gamma_limit=(90, 120), 
            p=0.3
        ),
        A.Rotate(limit=15, p=0.3 , fill_mask = 0),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        # A.Lambda(image=normalize_xca),
        ]
    )
    ds = TrainUnetDataset(transform=train_transforms,data = [[np.random.rand(512,512),np.random.rand(512,512)],[np.random.rand(512,512),np.random.rand(512,512)]])
    dl = DataLoader(ds,batch_size=2)
    for img , masks in dl:
        print(img.shape)
        for mask in masks:
            print(mask.shape)

    