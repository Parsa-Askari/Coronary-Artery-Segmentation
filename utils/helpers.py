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
import zarr
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize
import seaborn as sns
import json
###IE###
from .dataset import UnetExampleDataset , ValidUnetDataset
###SS###
def read_images(base_path, part,preprocessor,max_workers=None,chosen_labels = None):
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

        return img, label , name_stem

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(32, cpu * 4)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for img, label , name_stem in tqdm(ex.map(_read_one, image_names), total=len(image_names)):
            results.append([img,label,name_stem])

    return results

@torch.no_grad()
def make_example_datasets(valid_images,image_names,transform):
    img_dict = {"easy":[],"normal":[],"hard":[],"very hard":[]}
    for img, label , name_stem in valid_images : 
        for diff in image_names:
            if(name_stem in image_names[diff]):
                img_dict[diff].append([img , label , name_stem])
                break
    for diff , images in img_dict.items():
        ds = ValidUnetDataset(
            transform = transform,
            data = images
        )
        dl = DataLoader(
            ds,
            batch_size=1,
            num_workers=0,
            shuffle=False
        )
        img_dict[diff] = dl

    return img_dict

@torch.no_grad()
def no_teacher_forcing_pipeline(model,contexts,targets,class_count,device):
    """
    contexts = B x 2 x H x W
    targets = B x T x H x W
    """
    t_max = targets.shape[1]
    b_size = targets.shape[0]
    full_target_preds = []
    contexts = contexts.to(device)
    for i in range(t_max):
        pred_target = model(contexts) # B x 26 x H x W
        full_target_preds.append(pred_target)
        pred_target = torch.argmax(pred_target,dim=1) # B x H x W
        contexts[:,0] = pred_target

    full_target_preds = torch.stack(full_target_preds,dim=0) # T x B x 26 x H x W

    full_target_preds = full_target_preds.permute(1,0,2,3,4) # B x T x 26 x H x W
    
    return full_target_preds, targets.to(device) , b_size , t_max

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
def process_targets(pred_targets , targets , b_size , t_max , target_shape , class_count):
    """
    pred_targets = B x T , 26 , H , W
    targets = B x T , H , W 
    """
    pred_targets = torch.argmax(pred_targets,dim=1) # BxT , H, W
    pred_targets = pred_targets.view(b_size,t_max,*target_shape) # B , T , H, W
    pred_targets = pred_targets.max(dim=1).values # B , H, W
    pred_targets = F.one_hot(pred_targets, num_classes=class_count) # B , H , W , 26
    pred_targets = pred_targets.permute(0, 3, 1, 2).float()  # B , 26 , H , W 

    targets = targets.view(b_size,t_max,*target_shape) # B , T , H, W
    targets = targets.max(dim=1).values # B , H , W

    return pred_targets , targets
@torch.no_grad()
def TP_TN_FP_FN(preds,gt,process_preds=False,return_TN=False):
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
    img = image.copy().astype(np.uint8)
    m = mask.astype(np.int64)
    if(colors is None):
        colors = np.array([(0,255,0)]*25,dtype=np.uint8)
    img[m>0] = colors[m[m>0]-1]
    return img

@torch.no_grad()
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
        new_img = draw_mask(new_img,new_mask[0].numpy())

        plt.subplot(w,h,i)
        plt.imshow(old_img,cmap="gray")
        plt.title("Old Image")

        plt.subplot(w,h,i+1)
        plt.imshow(new_img)
        plt.title("New Image")

def pre_hard_skeletonize(base_path,output_path):
    parts = ["train","val","test"]
    os.makedirs(os.path.join(output_path , "skels"),exist_ok=True)
    for part in parts:
        mask_base_path = os.path.join(base_path,"labels",part)
        os.makedirs(os.path.join(output_path , "skels",part),exist_ok=True)

        mask_list = os.listdir(mask_base_path)
        for mask_name in tqdm(mask_list):
            name = Path(mask_name).stem
            mask_path = os.path.join(mask_base_path,mask_name)

            mask = zarr.load(str(mask_path))
       
            mask = (mask!=0).astype(np.uint8)
        
            out_skel_path = os.path.join(output_path,"skels",part,f"{name}.png")
            skel = skeletonize(mask).astype(np.uint8) * 255
            cv2.imwrite(out_skel_path,skel)
@torch.no_grad()
def pre_soft_skeletonize(base_path,output_path,batch_size=10,k=25):
    parts = ["train","val","test"]
    os.makedirs(os.path.join(output_path , "skels_soft"),exist_ok=True)
    for part in parts:
        mask_base_path = os.path.join(base_path,"labels",part)
        os.makedirs(os.path.join(output_path , "skels_soft",part),exist_ok=True)

        mask_list = os.listdir(mask_base_path)
        mask_buffer = []
        name_buffer = []
        for i,mask_name in enumerate(tqdm(mask_list)):
            name = Path(mask_name).stem
            mask_path = os.path.join(mask_base_path,mask_name)
            mask = zarr.load(str(mask_path))
            mask = (mask!=0).astype(np.float32)

            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            mask_buffer.append(mask)
            name_buffer.append(name)
            if((i+1)%batch_size==0 or i==len(mask_list)-1):
                mask_buffer = torch.cat(mask_buffer,dim=0).to("cuda")
                skels = soft_skeletonize(mask_buffer,k=k)
                skels = skels.cpu().numpy()
                B = skels.shape[0]
                for i in range(B):
                    skel = skels[i,0].astype(np.uint8)*255
                    o_name = name_buffer[i]
                    out_skel_path = os.path.join(
                        output_path,"skels_soft",part,f"{o_name}.png")
                    cv2.imwrite(out_skel_path,skel)
                mask_buffer = []
                name_buffer = []

@torch.no_grad()
def compute_confution_matrix(data_loader,model,class_maps,output_folder_path=None,draw_plot = True,class_count=26):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf_mat = torch.zeros((class_count,class_count))
    model.eval()
    for contexts , targets in tqdm(data_loader):
        with torch.autocast(device_type=device,dtype=torch.float16):
            pred_targets , targets ,b_size , t_max= no_teacher_forcing_pipeline(
                model = model,
                contexts = contexts,
                targets = targets,
                class_count = class_count,
                device = device
            )
            # pred_targets : B x T x 26 x H x W
            # targets : B x T x H x W
        pred_targets = pred_targets.reshape(-1,class_count,targets.shape[2],targets.shape[3])
        targets = targets.reshape(-1,targets.shape[2],targets.shape[3])
        pred_targets , targets = process_targets(
            pred_targets=pred_targets,
            targets= targets,
            b_size=b_size,
            t_max=t_max,
            target_shape=(targets.shape[1],targets.shape[2]),
            class_count=class_count
        )
        # pred_targets = B , 26 , H , W 
        # targets : B x H x W
        targets = targets.reshape(-1)
        pred_targets = torch.argmax(pred_targets,dim=1).view(-1) # B x H x W
        encoded_results = (targets*class_count + pred_targets).cpu() # # B x H x W
        counts = torch.bincount(encoded_results,minlength=class_count**2).view(class_count,class_count)
        conf_mat += counts
        
    conf_mat = conf_mat.float() / conf_mat.sum(dim=1,keepdims=True).clamp(min=1)
    conf_mat = conf_mat.numpy()
    if(draw_plot):
        class_names = ["background" for i in range(class_count)]
        for index , name in class_maps.items():
            class_names[index] = name
        plt.figure(figsize=(20,20))
        ax = sns.heatmap(
            conf_mat,
            annot=True,
            fmt=".2f",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues"
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        if(output_folder_path):
            out_path = os.path.join(output_folder_path,"conf_mat.png")
            plt.savefig(out_path)
    return conf_mat

def erode(mask):
    h_pool = -F.max_pool2d(-mask,(3,1),(1,1),(1,0))
    v_pool = -F.max_pool2d(-mask,(1,3),(1,1),(0,1))
    return torch.min(v_pool,h_pool)
def dilate(mask):
    return F.max_pool2d(mask,(3,3),(1,1),(1,1))
def soft_open(mask):
    return dilate(erode(mask))
def soft_skeletonize(I,k=25):
    I_ = soft_open(I)
    S = F.relu(I-I_)
    for i in range(k):
        I = erode(I)
        I_ = soft_open(I)
        S = S + (1-S)*F.relu(I-I_)
    return S