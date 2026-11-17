from torch.utils.data import Dataset
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
from torch_geometric.utils import (
    dropout_edge,dropout_node,
    add_random_edge,mask_feature )
from torch_geometric.data import Data , DataLoader
import torch_geometric.data as tg_data
###IE###
from .preprocessing import(
    gnn_cord_jitter,noisy_feature_jitter,gnn_gaussian_noise,
    gnn_masking,gnn_edge_drop
)
###SS###

class TrainUnetDataset(Dataset):
    def __init__(self, transform,data,normalizer,example_phase=False,part="train"):
        super(TrainUnetDataset, self).__init__()
        self.data = data
        self.transform = transform
        self.normalizer = normalizer
        self.example_phase = example_phase
        self.part = part
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        img : (H,W,C)
        maks : (H,W)
        """
        outs = self.data[index]

        img = outs["img"]
        hard_idx = outs["hard_idx"]
        mask = outs["label"]
        skel_mask = outs["skel_mask"]
        ignore_mask = outs["ignore_mask"]
        side = outs["side"]
        name = outs["name_stem"]

        small_vmask = np.zeros_like(mask)
        small_vmask[hard_idx[0],hard_idx[1]] = 1


        if img.ndim == 2: img = img[..., None]
        if mask.ndim == 2: 
            mask = mask[..., None]
            ignore_mask = ignore_mask[...,None]
            skel_mask = skel_mask[...,None]
            
            # expert_mask = expert_mask[...,None]
        # if v_mask.ndim == 2: v_mask = v_mask[...,None]


        result = self.transform(
            image=img,
            mask=mask,
            v_mask=ignore_mask,
            expert_mask=skel_mask,
            # topology_mask = topology_mask
        )
        
        new_image_not_normal  = result['image']
        new_mask= result['mask']
        new_small_vmask= result['v_mask']
        new_skel_mask = result["expert_mask"]
        
        # new_topology_mask = result["topology_mask"]
        # new_v_mask = result["v_mask"]
        # new_expert_mask = result["expert_mask"]

        new_image = self.normalizer(image = new_image_not_normal)["image"]

        new_mask = torch.from_numpy(new_mask).squeeze(-1).long() #(H,W)
        new_small_vmask = torch.from_numpy(new_small_vmask).squeeze(-1).long()
        new_skel_mask = torch.from_numpy(new_skel_mask).squeeze(-1).long() #(H,W)
        # new_topology_mask = torch.from_numpy(new_topology_mask).squeeze(-1).long()
        # binary_mask = (multi_mask !=0).squeeze(-1).long() #(H,W)
        # new_v_mask = torch.from_numpy(new_v_mask).squeeze(-1)
        # new_expert_mask = torch.from_numpy(new_expert_mask).squeeze(-1)
        # new_v_mask = torch.clamp(new_v_mask, min=0.0)
        
        if torch.is_tensor(new_image): new_image = new_image.numpy()
        
        # turn image shape to : (C,H,W)
        if new_image.ndim == 3 : 
            new_image = new_image.transpose(2, 0, 1) 
        elif new_image.ndim == 2:
            new_image = new_image[None, :, :] 

        
        new_image = new_image.astype(np.float32)
        
        if(self.example_phase):
            return {
                "new_image_not_normal" : new_image_not_normal,
                "imgs" : new_image, 
                "mask" : new_mask ,
                "skel_mask":new_skel_mask,
                "small_vmask":new_small_vmask,
                "side" : float(side),
                "name" : name}
        
        return {
            "imgs" : new_image, 
            "mask" : new_mask,
            "small_vmask" : new_small_vmask,
            # "topology_mask" : new_topology_mask,
            "skel_mask":new_skel_mask,
            "side" : float(side),
            "name" : name
        }

class GnnDataset(tg_data.Dataset):
    def __init__(self,data,part,family,training_mode="multi",
                 transform=None,aug_args=None,eval_mode=False):
        super().__init__()
        
        self.part = part
        self.training_mode = training_mode
        self.transform = transform
        self.aug_args = aug_args
        if(aug_args is not None):
            self.cont_idx = torch.tensor(self.aug_args["cont_idx"])
            self.masking_idx = torch.tensor(self.aug_args["masking_idx"])
            self.cord_idx = torch.tensor(self.aug_args["cord_idx"])

        self.data = [{**d["correct"],"name":d["name"]} for d in data if family in d["correct"]["features"]]
        # for d in self.data:
        #     if(d["name"] == "133"):
        #         print(d["features"][family])
        #         print("=====")
        #         print(d["edges"][family])
        #         print("=====")
        #         print(d["labels"][family])
        #         print("=====")
        data_size = len(self.data)
        self.family = family
        self.eval_mode = eval_mode
        print(f"total of {data_size} for family {family}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        batch = self.data[index]

        features  = batch["features"][self.family]
        weights = batch["weights"][self.family]
        edge_index = batch["edges"][self.family]
        labels = batch["labels"][self.family]
        name = batch["name"]
        # print(name,features.shape)
        features = torch.from_numpy(features).float().clone()
        edge_index = torch.from_numpy(edge_index).int().clone()
        labels = torch.from_numpy(labels).long().clone()
        weights = torch.from_numpy(weights).float().clone()

        if(not self.eval_mode and self.part == "train" and self.training_mode!="side"):
            if(random.random()<0.5 and len(self.cord_idx)!=0):
                features[...,self.cord_idx] = gnn_cord_jitter(
                    cords = features[...,self.cord_idx],
                    cord_smoother=self.aug_args["cord_smoother"])
            if(random.random()<0.5 and len(self.cont_idx)!=0):
                features[...,self.cont_idx] = gnn_gaussian_noise(
                    features[...,self.cont_idx],
                    g_smoother = self.aug_args["g_smoother"]
                )
            features = noisy_feature_jitter(
                features = features,
                prob_idx = self.aug_args["prob_idx"],
                radius_idx = self.aug_args["radius_idx"],
                frangi_idx = self.aug_args["frangi_idx"],
                prob_smoother = self.aug_args["prob_smoother"],
                radius_smoother = self.aug_args["radius_smoother"],
                frangi_smoother = self.aug_args["frangi_smoother"]
            )
            if(random.random()<0.5 and len(self.masking_idx)!=0):
                features[...,self.masking_idx] = gnn_masking(
                    features[...,self.masking_idx],
                    mask_prob = self.aug_args["mask_prob"]
                )
            if(random.random()<0.5):
                edge_index = gnn_edge_drop(
                    edge_idx = edge_index,
                    e_prob = self.aug_args["e_prob"]
                )
        data = Data(x=features,edge_index=edge_index,y=labels,node_weights=weights)
        return data ,1,name


class UnetExampleDataset(Dataset):
    def __init__(self,transform,data,normalizer,loss_check=False,gnn_ds=False):
        super(UnetExampleDataset,self).__init__()
        
        self.data = data
        self.transform = transform
        self.normalizer = normalizer
        self.loss_check = loss_check
        self.gnn_ds = gnn_ds
        self.left_side = [1,2,3,4,20,21,22,23]
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        outs = self.data[index]

        img = outs["img"]
        hard_idx = outs["hard_idx"]
        mask = outs["label"]
        skel_mask = outs["skel_mask"]
        ignore_mask = outs["ignore_mask"]
        side = outs["side"]
        name = outs["name_stem"]\
        
        small_vmask = np.zeros_like(mask)
        small_vmask[hard_idx[0],hard_idx[1]] = 1

        if img.ndim == 2: img = img[..., None]
        if mask.ndim == 2: 
            mask = mask[..., None]
            ignore_mask = ignore_mask[...,None]
            skel_mask = skel_mask[...,None]

        
            # expert_mask = expert_mask[...,None]
        # if v_mask.ndim == 2: v_mask = v_mask[...,None]
        
        result = self.transform(
            image=img,
            mask=mask,
            v_mask=ignore_mask,
            expert_mask=skel_mask
        )
        new_image_not_normal = result['image']
        new_mask = result["mask"]
        new_small_vmask = result["v_mask"]
        new_skel_mask = result["expert_mask"]

        new_image = self.normalizer(image = new_image_not_normal)["image"]

        new_image = new_image.transpose(2, 0, 1) 
        new_image_not_normal = new_image_not_normal.transpose(2,0,1)

        new_mask = new_mask[:,:,0]
        new_small_vmask = new_small_vmask[...,0]
        new_skel_mask = new_skel_mask[...,0]

        u = np.unique(new_mask)[1:]
        if(u[0] in self.left_side):
            side = 0
        else:
            side = 1
        binary_mask = (new_mask!=0).astype(int)

        data = {
            "new_image_not_normal" : new_image_not_normal , 
            "new_image" : new_image ,
            "name" : name}
        
        if(self.loss_check):
            data["mask"] = binary_mask
            return data
        if(self.gnn_ds):
            data["mask"] = new_mask
            data["side"] = side
            data["small_vmask"] = new_small_vmask
            data["skel_mask"] = new_skel_mask
            return data
        return {
            "new_image" : new_image, 
            "name" : name}

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

    