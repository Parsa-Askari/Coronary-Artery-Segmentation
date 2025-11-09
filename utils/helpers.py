import os
import zarr
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
###IE###
from .dataset import UnetExampleDataset
def read_images(base_path, part,preprocessor, max_workers=None):
    base_path = Path(base_path)
    images_base = base_path / "images" / part
    labels_base = base_path / "labels" / part

    image_names = sorted([p.name for p in os.scandir(images_base) if p.is_file()])
    if(not preprocessor):
        print("NOTE : preprocessor is not defined . no preprocessing will be used !")
    def _read_one(fname):
        name_stem = Path(fname).stem
        img_path = images_base / fname
        label_path = labels_base / f"{name_stem}.zarr"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if(preprocessor):
            img = preprocessor(img)
        label = zarr.load(str(label_path))
        return img, label

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(32, cpu * 4)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for img, label in tqdm(ex.map(_read_one, image_names), total=len(image_names)):
            results.append([img, label])

    return results

def to_device(img,gt_mask,device,binary_mode):
    gt_mask = gt_mask.long()
    img = img.to(device)
    gt_mask = gt_mask.to(device)
    if(binary_mode):
        gt_label = gt_label.to(device)
    else :
        gt_label = None
    return img , gt_mask 


def crop_dims(target , current):
    left = (current.shape[3]-target.shape[3])//2
    right = (current.shape[3]-target.shape[3]) - left
    top = (current.shape[2]-target.shape[2])//2
    down = (current.shape[2]-target.shape[2]) - top
    croped = current[:,:,top:-down , left:-right]
    return croped
def padd_dims(target , current):
    pad_h = target.shape[2] - current.shape[2] 
    pad_w = target.shape[3] - current.shape[3]
    padded = F.pad(current, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return padded

@torch.no_grad()
def TP_TN_FP_FN(preds,gt,process_preds=True,return_TN=False):
    if(process_preds):
        preds_argmax = torch.argmax(preds,dim=1)
        onehot_preds = F.one_hot(preds_argmax,num_classes=preds.shape[1])
        pred_onehot = onehot_preds.permute(0, 3, 1, 2).float()
    else :
        pred_onehot = preds
        
    onehot_gt = F.one_hot(gt,num_classes=preds.shape[1])
    onehot_gt = onehot_gt.permute(0, 3, 1, 2).float()
    TN = 0
    if(return_TN):
        TN = (((1-onehot_gt)*(1-pred_onehot)).sum(dim=(0,2,3))).cpu()
    TP = ((onehot_gt*pred_onehot).sum(dim=(0,2,3))).cpu()
    
    FP = (((1-onehot_gt)*pred_onehot).sum(dim=(0,2,3))).cpu()
    FN = ((onehot_gt*(1-pred_onehot)).sum(dim=(0,2,3))).cpu()
    return TP , TN , FP , FN

def draw_mask(image,mask,args=None,colors=None):
    img = image.copy()
    if(args is not None):
        class_count= args["class_count"]
    H,W,C=img.shape
    for i in range(H):
        for j in range(W):
            c = mask[i,j]
            if(c==0):
                continue
            if(colors is not None):
                img[i,j] = colors[c-1]
            else :
                img[i,j] = (0,255,0)
    # plt.imshow(image)
    return img.astype(np.uint8)

def plot_some_images(data,transforms,image_counts=36,fig_shape=(6,6),base_transforms=None):
    ds = UnetExampleDataset(transform=transforms , data=data,base_transform=base_transforms)
    dataloader = DataLoader(
        ds,
        batch_size = 2 ,
        num_workers = 4 ,
        pin_memory=False,
        shuffle=True
    )

    iter_loader = iter(dataloader)
    w,h=fig_shape
    plt.figure(figsize=(w*5,h*5))
    for i in range(1,image_counts+1,2):
        new_imgs , new_mask , old_imgs , old_mask = next(iter_loader)
        new_img = new_imgs[0].numpy()
        old_img = old_imgs[0].numpy()
        if(new_img.shape[0]==1):
            new_img = new_img[0]
            old_img = old_img[0]

        x_disp = (new_img- new_img.min()) / (new_img.max() - new_img.min() + 1e-8)
        new_img = np.repeat(x_disp[..., None], 3, axis=2)*255
        new_img = draw_mask(new_img,new_mask[0])

        plt.subplot(w,h,i)
        plt.imshow(old_img,cmap="gray")
        plt.title("Old Image")

        plt.subplot(w,h,i+1)
        plt.imshow(new_img)
        plt.title("New Image")

        
