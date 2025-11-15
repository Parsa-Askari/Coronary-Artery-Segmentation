from torch.utils.data import Dataset,DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
###IE###
###SS###
class UnetDataset(Dataset):
    def __init__(self,transform,data,base_size=512,out_counts=7):
        super(UnetDataset,self).__init__()
        self.data = data
        self.transform = transform
        self.to_tensor = ToTensorV2()
        self.resizers = [
            A.Resize(base_size//(2**i),base_size//(2**i),
            interpolation=cv2.INTER_NEAREST,
            mask_interpolation=cv2.INTER_NEAREST) for i in range(1,out_counts)
        ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img , mask = self.data[index]
        img = np.expand_dims(img, axis=-1) 
        mask = mask[...,None]
        result = self.transform(image=img, mask=mask)
        new_image = result['image']
        new_mask = result['mask'].squeeze(-1)

        new_masks =[new_mask] + [resizer(image = new_mask)["image"] for resizer in self.resizers]


        new_image = self.to_tensor(image = new_image)["image"]
        new_masks = [torch.tensor(m).long() for m in new_masks]
        return new_image.float() , new_masks

class ValidUnetDataset(Dataset):
    def __init__(self,transform,data):
        super(ValidUnetDataset,self).__init__()
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img , mask = self.data[index]
        img = np.expand_dims(img, axis=-1) 
        mask = mask[...,None]
        result = self.transform(image=img, mask=mask)
        new_image = result['image']
        new_mask = result['mask'].squeeze(-1)

        return new_image.float() , [new_mask.long()]

class UnetExampleDataset(Dataset):
    def __init__(self,transform,data,base_transform=None):
        super(UnetExampleDataset,self).__init__()
        
        self.data = data
        self.transform = transform
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
    ds = UnetDataset(transform=train_transforms,data = [[np.random.rand(512,512),np.random.rand(512,512)],[np.random.rand(512,512),np.random.rand(512,512)]])
    dl = DataLoader(ds,batch_size=2)
    for img , masks in dl:
        print(img.shape)
        for mask in masks:
            print(mask.shape)

    