from torch.utils.data import Dataset,DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import cv2
import torch
###IE###
###SS###
class UnetDataset(Dataset):
    def __init__(self,transform,data,base_size=512,valid=False):
        super(UnetDataset,self).__init__()
        self.data = data
        self.transform = transform
        self.to_tensor = ToTensorV2()
        self.base_size=base_size
        self.valid=valid
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img,side_label,binary_mask,abs_mask,mask = self.data[index]
        # img = np.expand_dims(img, axis=-1) 
        mask = mask[...,None]
        binary_mask = binary_mask[...,None]
        abs_mask = abs_mask[...,None]
        result = self.transform(
            image=img,
            mask=mask,
            binary_mask = binary_mask,
            abs_mask = abs_mask,
        )

        new_image = self.to_tensor(image = result['image'])["image"]
        new_mask = self.to_tensor(mask = result['mask'])["mask"].squeeze(-1)
        new_abs_mask = self.to_tensor(mask = result["abs_mask"])["mask"].squeeze(-1)
    
        new_binary_mask = result["binary_mask"]
        

        new_binary_mask = self.to_tensor(mask = new_binary_mask)["mask"].squeeze(-1)

        new_binary_mask = new_binary_mask.long()
        new_abs_mask = new_abs_mask.long()
        new_mask = new_mask.long()
        # new_image = self.to_tensor(image = new_image)["image"]
        return (
            new_image.float() , 
            torch.LongTensor([side_label]),
            new_binary_mask.unsqueeze(0),
            new_abs_mask,
            new_mask
        )

class ValidUnetDataset(Dataset):
    def __init__(self,transform,data):
        super(ValidUnetDataset,self).__init__()
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img,side_label,binary_mask,abs_mask,mask = self.data[index]
        # img = np.expand_dims(img, axis=-1) 
        mask = mask[...,None]
        binary_mask = binary_mask[...,None]
        abs_mask = abs_mask[...,None]

        result = self.transform(
            image=img,
            mask=mask,
            binary_mask = binary_mask,
            abs_mask = abs_mask
        )
        new_image = result['image']
        new_mask = result['mask'].squeeze(-1)
        new_abs_mask = result["abs_mask"].squeeze(-1)
        new_binary_mask = result["binary_mask"].squeeze(-1)

        new_binary_mask = new_binary_mask.long()
        new_abs_mask = new_abs_mask.long()
        new_mask = new_mask.long()

        # new_image = self.to_tensor(image = new_image)["image"]
        return new_image.float() , side_label ,new_binary_mask , new_abs_mask , new_mask

class UnetExampleDataset(Dataset):
    def __init__(self,transform,data,base_transform=None):
        super(UnetExampleDataset,self).__init__()
        
        self.data = data
        self.transform = transform
        self.to_tensor = ToTensorV2()
        if(base_transform is None):
            self.base_transform = A.Compose(
                [ToTensorV2()]
            )
        else:
            self.base_transform = base_transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img,side_label,binary_mask,abs_mask,mask = self.data[index]
        # img = np.expand_dims(img, axis=-1) 
        # print(img.shape)
        mask = mask[...,None]
        binary_mask = binary_mask[...,None]
        abs_mask = abs_mask[...,None]
        
        result = self.transform(
            image=img,
            mask=mask,
            binary_mask = binary_mask,
            abs_mask = abs_mask
        )
        new_image = result['image']
        new_mask = result['mask'].squeeze(-1)
        new_abs_mask = result["abs_mask"].squeeze(-1)
        new_binary_mask = result["binary_mask"].squeeze(-1)
        
        raw_result = self.base_transform(
            image=img,
            mask=mask,
            binary_mask = binary_mask,
            abs_mask = abs_mask
        )
        raw_image = raw_result['image']
        raw_mask = raw_result['mask'].squeeze(-1)
        raw_abs_mask = raw_result["abs_mask"].squeeze(-1)
        raw_binary_mask = raw_result["binary_mask"].squeeze(-1)
        new_image = self.to_tensor(image = new_image)["image"]
        return new_image.float() , new_mask  , raw_image.float() , raw_mask

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

    