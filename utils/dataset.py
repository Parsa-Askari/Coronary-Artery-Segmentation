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
def collate_fn_train(batch):
    lengths = [x[2].shape[0] for x in batch]
    max_t = max(lengths)
    batch_size = len(batch)
    
    _, H, W = batch[0][0].shape 
    
    batch_inputs = torch.zeros((batch_size, max_t, 2, H, W), dtype=torch.float32)
    batch_targets = torch.zeros((batch_size, max_t, H, W), dtype=torch.long)

    for i, (img, ctx, tgt) in enumerate(batch):
        t = lengths[i]
        
        img_t = torch.from_numpy(img)
        ctx_t = torch.from_numpy(ctx)
        tgt_t = torch.from_numpy(tgt)

        batch_inputs[i, :t, 0] = ctx_t
        batch_inputs[i, :t, 1] = img_t 
        batch_targets[i, :t] = tgt_t
        
        if t < max_t:
            batch_inputs[i, t:, 0] = ctx_t[-1]
            batch_inputs[i, t:, 1] = img_t

    return batch_inputs, batch_targets
    
def collate_fn_test(batch):
    lengths = [x[1].shape[0] for x in batch]
    max_t = max(lengths)
    batch_size = len(batch)
    
    _, H, W = batch[0][0].shape 
    
    batch_targets = torch.zeros((batch_size, max_t, H, W), dtype=torch.long)
    batch_inputs = torch.zeros((batch_size, 2, H, W), dtype=torch.float32)
    for i, (ctx, tgt) in enumerate(batch):
        t = lengths[i]
        tgt_t = torch.from_numpy(tgt)
        batch_targets[i, :t] = tgt_t
        batch_inputs[i] = ctx

    return batch_inputs, batch_targets

class TrainUnetDataset(Dataset):
    def __init__(self, transform, data):
        super(TrainUnetDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, mask,_ = self.data[index]
        
        if img.ndim == 2: img = img[..., None]
        if mask.ndim == 2: mask = mask[..., None]

        result = self.transform(image=img, mask=mask)
        new_image = result['image']
        new_mask = result['mask']

        if torch.is_tensor(new_image): new_image = new_image.numpy()
        if torch.is_tensor(new_mask): new_mask = new_mask.numpy()
        
        if new_image.ndim == 3 and new_image.shape[-1] == 1: 
            new_image = new_image.transpose(2, 0, 1)
        elif new_image.ndim == 2:
            new_image = new_image[None, :, :]
            
        if new_mask.ndim == 3: new_mask = new_mask.squeeze(-1)

        u = np.unique(new_mask)
        labels = u[1:] if u[0] == 0 else u
        targets = (new_mask[None, :, :] == labels[:, None, None]).astype(np.float32) * labels[:, None, None]

        accumulated = np.cumsum(targets, axis=0)
        zeros = np.zeros((1, new_mask.shape[0], new_mask.shape[1]), dtype=np.float32)

        if len(labels) > 1:
            contexts = np.concatenate([zeros, accumulated[:-1]], axis=0)
        else:
            contexts = zeros

        return new_image, contexts, targets

class ValidUnetDataset(Dataset):
    def __init__(self, transform, data):
        super(ValidUnetDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, mask,_ = self.data[index]
        
        if img.ndim == 2: img = img[..., None]
        if mask.ndim == 2: mask = mask[..., None]

        result = self.transform(image=img, mask=mask)
        new_image = result['image']
        new_mask = result['mask']

        if torch.is_tensor(new_image): new_image = new_image.numpy()
        if torch.is_tensor(new_mask): new_mask = new_mask.numpy()
        
        if new_image.ndim == 3 and new_image.shape[-1] == 1: 
            new_image = new_image.transpose(2, 0, 1)
        elif new_image.ndim == 2:
            new_image = new_image[None, :, :]
            
        if new_mask.ndim == 3: new_mask = new_mask.squeeze(-1)

        u = np.unique(new_mask)
        labels = u[1:] if u[0] == 0 else u

        targets = (new_mask[None, :, :] == labels[:, None, None]).astype(np.float32) * labels[:, None, None]


        contexts = np.zeros((1, new_mask.shape[0], new_mask.shape[1]), dtype=np.float32)
        contexts = np.concatenate([contexts,new_image],axis=0)
        contexts = torch.from_numpy(contexts)
        return contexts , targets

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

    