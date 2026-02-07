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
    """
    batch : list of batches [img , targets , stops] 
        -   img : (C,H,W)
        -   targets : (unique_len,H,W)
        -   stops : (unique_len)
        -   full_masks : (H,W)
    """
    lengths = [x[2].shape[0] for x in batch]
    max_t = max(lengths)
    batch_size = len(batch)
    
    H, W, C = batch[0][0].shape 
    
    batch_imgs = torch.zeros((batch_size,C,H, W), dtype=torch.float32)
    batch_targets = torch.zeros((batch_size, max_t, H, W), dtype=torch.long)
    batch_stops = torch.zeros((batch_size, max_t), dtype=torch.long)
    batch_full_masks = torch.zeros((batch_size, H, W), dtype=torch.long)

    for i, (img, target, stop_label , full_mask) in enumerate(batch):
        t = lengths[i]
        
        img_t = torch.from_numpy(img).permute(2,0,1)
        target_t = torch.from_numpy(target)
        stop_label_t = torch.from_numpy(stop_label)
        full_mask_t = torch.from_numpy(full_mask)
        
        batch_imgs[i] = img_t
        batch_targets[i, :t] = target_t 
        batch_stops[i, :t] = stop_label_t
        batch_full_masks[i] = full_mask_t

    """
    batch_imgs : (B,C,H,W)
    batch_targets : (B,seq_len,H,W)
    batch_size : (B,seq_len)
    batch_full_masks : (B,H,W)
    """
    return batch_imgs, batch_targets, batch_stops, batch_full_masks

class TrainUnetDataset(Dataset):
    def __init__(self, transform, data):
        super(TrainUnetDataset, self).__init__()
        self.data = data
        self.transform = transform

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
        new_image = result['image']
        new_mask = result['mask']

        if torch.is_tensor(new_image): new_image = new_image.numpy()
        if torch.is_tensor(new_mask): new_mask = new_mask.numpy()
        
        # turn image shape to : (C,H,W)
        if new_image.ndim == 3 and new_image.shape[-1] == 1: 
            new_image = new_image.transpose(2, 0, 1) 
        elif new_image.ndim == 2:
            new_image = new_image[None, :, :] 
        # mask shape to : (H,W)
        if new_mask.ndim == 3: new_mask = new_mask.squeeze(-1)

        u = np.unique(new_mask)
        labels = u[1:] if u[0] == 0 else u
        # targets : (unique_len,H,W)
        targets = (new_mask[None, :, :] == labels[:, None, None]).astype(np.float32) * labels[:, None, None]
        # NOTE : we add an stop token here also with the shape (1,H,W) full zero
        stop_token = np.zeros((1,new_mask.shape[0],new_mask.shape[1]),dtype=np.float32)
        targets = np.concatenate([targets,stop_token],axis=0)
        # stop labels = (unique_len+1)
        stop_labels = np.ones(len(labels)+1)
        stop_labels[-1] = 0
        return new_image, targets, stop_labels ,new_mask


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

    