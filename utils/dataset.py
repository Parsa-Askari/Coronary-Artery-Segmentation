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
from .preprocessing import crop_with_bbox
###SS###
class ToTensor:
    def __init__(slef):
        pass
    def __call__(self,x):
        if(len(x.shape)==2):
            x = x[...,None]
        x = torch.from_numpy(x)
        x = x.permute(2,0,1)
        return x

class GroupPickSampler(Sampler):
    def __init__(self, group_to_idxs, seed=0, extras_per_group=1):
        self.group_to_idxs = group_to_idxs
        self.seed = seed
        self.epoch = 0
        self.extras_per_group = extras_per_group

        self._len = 0
        for idxs in group_to_idxs:
            self._len += 1 
            self._len += min(extras_per_group, max(0, len(idxs)-1))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        order = torch.randperm(len(self.group_to_idxs), generator=g).tolist()

        for i in order:
            idxs = self.group_to_idxs[i]

            yield idxs[0]  # anchor

            rest = idxs[1:]
            if rest:
                k = min(self.extras_per_group, len(rest))
                pick = torch.randperm(len(rest), generator=g)[:k].tolist()
                for j in pick:
                    yield rest[j]

    def __len__(self):
        return self._len
    
class UnetDataset(Dataset):
    def __init__(self,transform,data,crop_prob=0.33,base_size=512,
                 valid=False,out_counts=1,just_binary_trining=True,binary_type="both"):
        super(UnetDataset,self).__init__()
        self.data = data
        self.transform = transform
        self.base_size=base_size
        self.valid=valid
        self.crop_prob = crop_prob
        self.just_binary_trining=just_binary_trining
        self.to_tensor = ToTensor()
        self.dsv_transforms = [
            A.Resize(
                base_size[0]//(2**i), 
                base_size[1]//(2**i), 
                interpolation=cv2.INTER_NEAREST
            ) for i in range(1,out_counts)
        ]
        self.binary_type = binary_type
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img, multi_mask ,binary_mask,bbox,name_stem= self.data[index]
        if(not self.valid):
            if(random.random()<self.crop_prob ):
                img, multi_mask ,binary_mask= crop_with_bbox(
                    img, multi_mask ,  binary_mask,bbox
                )
        # img = np.expand_dims(img, axis=-1) 
        multi_mask = multi_mask[...,None]
        binary_mask = binary_mask[...,None]
        result = self.transform(
            image=img,
            mask=multi_mask,
            binary_mask = binary_mask
        )
        
        new_image = result['image']
        new_multi_mask = result["mask"]
        new_binary_mask = result["binary_mask"]

        masks = []
        if(self.just_binary_trining):
            mask = new_binary_mask
        else:
            mask = new_multi_mask
        for i,resizer in enumerate(self.dsv_transforms):
            resized = self.to_tensor(
                resizer(image=mask)["image"]
            ).long()
            # print(resized.shape)
            masks.append(resized)
        
        masks = [
            self.to_tensor(mask).long()
        ] + masks
        new_image = self.to_tensor(new_image)
        # new_image = self.to_tensor(image = new_image)["image"]
        return (
            new_image.float() ,
            masks
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
    
def make_dataloader(data,args,valid=False,sampler_weights=None,sampler=None):
    if(sampler_weights is not None):
        print("using weighted sampler here")
        sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights))
        dataloader = DataLoader(
            data,
            batch_size = args["batch_size"] ,
            num_workers = args["num_workers"] ,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )
    elif(sampler is not None):
        print("using costum sampler here")
        dataloader = DataLoader(
            data,
            batch_size = args["batch_size"] ,
            num_workers = args["num_workers"] ,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )
    else : 
        if(valid):
            print("valid with no sampler")
            dataloader = DataLoader(
                data,
                batch_size = args["batch_size"] ,
                num_workers = args["num_workers"] ,
                pin_memory=True,
                shuffle=False,
            )
        else : 
            print("train with no sampler")
            dataloader = DataLoader(
                data,
                batch_size = args["batch_size"] ,
                num_workers = args["num_workers"] ,
                pin_memory=True,
                shuffle=True
            )
    return dataloader
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

    