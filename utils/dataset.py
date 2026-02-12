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
    batch : list of batches [img, mask , m_labels, m_taken] 
        -   img : (C,H,W)
        -   mask : (H,W)
        -   m_labels : (unique_len,H,W)
        -   m_taken : (unique_len,H,W)
    """
    lengths = [x[2].shape[0] for x in batch]
    max_t = max(lengths)+1
    batch_size = len(batch)
    
    C , H, W = batch[0][0].shape 
    
    batch_imgs = torch.zeros((batch_size,max_t,C,H, W), dtype=torch.float32)
    batch_masks = torch.zeros((batch_size, H, W), dtype=torch.long)
    batch_m_labels = torch.zeros((batch_size, max_t, H, W), dtype=torch.long)
    batch_m_taken = torch.zeros((batch_size, max_t, 1, H, W), dtype=torch.long)
    batch_current_label = torch.zeros((batch_size, max_t, 26), dtype=torch.long)

    for i, (img, mask , m_labels, m_taken) in enumerate(batch):
        t = lengths[i]
        u = np.unique(mask)[1:].astype(int)

        img_t = torch.from_numpy(img)
        mask_t = torch.from_numpy(mask)
        m_labels_t = torch.from_numpy(m_labels)
        m_taken_t = torch.from_numpy(m_taken)
        u_t = torch.from_numpy(u)

        batch_imgs[i,:] = img_t
        batch_masks[i] = mask_t 
        batch_m_labels[i, :t] = m_labels_t
        batch_m_taken[i, :t, 0] = m_taken_t

        batch_current_label[i,torch.arange(len(u)) ,u_t] = 1
        batch_current_label[i, u.shape[0] ,0] = 1
        
        batch_m_taken[i , t:] = m_taken_t[-1]

    # batch_current_label = torch.cumsum(batch_current_label,dim=1)
    """
    batch_imgs : (B,seq_len,C,H,W)
    batch_masks : (B, H, W)
    batch_m_labels : (B, seq_len, H, W)
    batch_m_taken : (B, seq_len, 1, H, W)
    batch_current_label : (B, max_t, 26)
    """
    return batch_imgs, batch_masks, batch_m_labels, batch_m_taken , batch_current_label

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
        if new_image.ndim == 3 : 
            new_image = new_image.transpose(2, 0, 1) 
        elif new_image.ndim == 2:
            new_image = new_image[None, :, :] 
        # mask shape to : (H,W)
        if new_mask.ndim == 3: new_mask = new_mask.squeeze(-1)

        u = np.unique(new_mask)
        labels = u[1:] if u[0] == 0 else u
        # m_labels : (unique_len,H,W)
        start_mask = np.zeros((1,new_image.shape[1],new_image.shape[2]))
        m_labels = (new_mask[None, :, :] == labels[:, None, None]).astype(np.float32) * labels[:, None, None]
        

        m_labels = np.concatenate([start_mask,m_labels],axis=0)
        # m_taken : (unique_len,H,W)
        m_taken = (np.cumsum(m_labels,axis=0)!=0).astype(int)


        return new_image, new_mask , m_labels, m_taken


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

    