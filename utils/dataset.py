from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
###IE###
class UnetDataset(Dataset):
    def __init__(self,transform,data):
        super(UnetDataset,self).__init__()
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        img , mask = self.data[index]
        img = np.expand_dims(img, axis=-1) 
        result = self.transform(image=img, mask=mask)
        new_image = result['image']
        new_mask = result['mask']

        return new_image.float() , new_mask

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
