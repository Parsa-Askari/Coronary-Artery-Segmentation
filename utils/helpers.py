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
from .dataset import UnetExampleDataset
###SS###
@torch.no_grad()
def read_images(base_path, part,preprocessor,in_c,resize_binary,
                max_workers=None,train_class_counts=None,k=40,just_binary_trining=False):
    base_path = Path(base_path)
    images_base = base_path / "images" / part
    labels_base = base_path / "labels" / part
    skels_base = base_path / "skels" / part
    transformed_base = base_path / "transformed" / part
    with open(f"data/{part}_side_labels.json","r") as f:
        side_labels = json.load(f)
    with open(f"data/{part}_fg_bboxes.json","r") as f:
        bboxes = json.load(f)
    if(train_class_counts is not None):
        freqs = train_class_counts / train_class_counts.sum()
        class_w = 1 / (freqs + 1e-6)
        class_w[0] = 0
    image_names = sorted([p.name for p in os.scandir(images_base) if p.is_file()])
    if(train_class_counts is not None):
        max_count = np.max(train_class_counts[1:])
        print("max count is : ",max_count)
    if(not preprocessor):
        print("NOTE : preprocessor is not defined . no preprocessing will be used !")
    def _read_one(fname):
        name_stem = Path(fname).stem
        img_path = images_base / fname
        label_path = labels_base / f"{name_stem}.zarr"
        t_img_path = transformed_base / f"{name_stem}.zarr"
        skel_path = skels_base / fname
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        # t_img = zarr.load(t_img_path)
        if(preprocessor):
            for p in preprocessor:
                img = p(img)

        if(in_c!=1):
            # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = np.stack([img, img, img], axis=-1)
            # t_img = np.stack([t_img, t_img, t_img], axis=-1)*255

        label = zarr.load(str(label_path))
        if(train_class_counts is not None):
            num_classes = class_w.shape[0]
            counts = np.bincount(label.reshape(-1), minlength=num_classes)
            weight = float((counts * class_w).sum())
        else:
            weight = None




        unique_labels = np.unique(label)
        unique_labels = unique_labels[unique_labels!=0]

        binary_label = (label!=0).astype(np.uint8)
        side_label = side_labels[name_stem]
        bbox = bboxes[name_stem]
        return img, label ,binary_label,bbox, weight , side_label

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(32, cpu * 4)

    results = []
    weights = []
    co=0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for img, label ,binary_label,bbox, weight , side_label  in tqdm(ex.map(_read_one, image_names), total=len(image_names)):
            results.append([img, label ,binary_label,bbox])
            weights.append(weight)
            if(just_binary_trining):
                if(side_label==0 and co!=250):
                    results.append([img, label ,binary_label,bbox])
                    weights.append(weight)
                    co+=1
    print(co)
    if(train_class_counts is not None):
        weights = torch.tensor(weights).double()
        weights +=1e-8
        weights /=weights.sum()
        return results , weights
    else:
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

def to_rgb(img):
    if(isinstance(img,np.ndarray)):
        img = torch.from_numpy(img)
    x_disp = (img- img.min()) / (img.max() - img.min() + 1e-8)
    
    img = torch.cat([x_disp,x_disp,x_disp],dim=0) *255
    # print("d",img.shape)
    img = img.permute(1,2,0)
    return img.numpy()
def denorm(img,mean,std):
    if(isinstance(img,np.ndarray)):
        img = torch.from_numpy(img)
    img = (img*std) + mean
    img = img.clamp(0,1)*255
    return img.permute(1,2,0).cpu().numpy().astype(np.uint8)

def draw_mask(image,mask,args=None,colors=None):
    img = image.copy().astype(np.uint8)
    m = mask.astype(np.int64)
    if(colors is None):
        colors = np.array([(0,255,0)]*25,dtype=np.uint8)
        c = np.array([(0,255,0)])
        
    else:
        
        c = colors[m[m>0]-1].reshape(-1,3)
        
    # print("----")
    # print(c.shape)
    # print(img[m>0].shape)
    img[m>0] = c
    # print(img.shape)
    return img

@torch.no_grad()
def plot_some_images(data,transforms,mean,std,image_counts=36,fig_shape=(6,6),
                     base_transforms=None):
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
   
        new_img = new_imgs[0]
        old_img = old_imgs[0]
        if(new_img.shape[0]==1):
            new_img = to_rgb(new_img)
            old_img = to_rgb(old_img)

        else:
            new_img = denorm(new_img,mean,std)
            old_img = denorm(old_img,mean,std)

        # print(old_img.min(),old_img.max())
        # print(new_img.min(),new_img.max())
        # print("--------------")
        new_img = draw_mask(new_img,new_mask[0].numpy())
        
        # print(old_img.min(),old_img.max())
        plt.subplot(w,h,i)
        plt.imshow(old_img)
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
def compute_confution_matrix(data_loader,model,class_maps,output_folder_path=None,draw_plot = True,class_count=26,use_amp=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf_mat = torch.zeros((class_count,class_count))
    model.eval()
    for img, masks in data_loader:
        img = img.to(device)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=use_amp):
            mask = masks[0].to(device).view(-1)
            pred_masks = model(img)[0]

        pred_mask = torch.argmax(pred_masks,dim=1).view(-1)
        encoded_results = (mask*class_count + pred_mask).cpu()
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
def labels_to_string(mask,remove_bg = True,max_size =13):
    s = 0
    if(remove_bg):
        s=1
    u = np.unique(mask)[1:].tolist()
    u = list(map(str,u))
    c = ""
    for i,label in enumerate(u) : 
        c+=label
        if((i+1)%max_size==0):
            c+="\n"
        else:
            c+="|"
    return c

def make_fg_bboxes(parts,base_path,space=10):
    for part in parts:
        print(f"processing {part}")
        main_path = os.path.join(base_path,f"{part}.json")
        out_path = os.path.join(base_path,f"{part}_fg_bboxes.json")
        ls={}

        with open(main_path , "r") as f:
            train_json = json.load(f)
        for path , data in tqdm(train_json.items()):
            name = Path(path).stem
            bbox_norm = data["bbox_norm"]
            x_min = float("inf")
            y_min = float("inf")
            x_max = 0
            y_max = 0
            for bbox in bbox_norm:
                x1,y1 = bbox[0:2]
                x2 = x1 + bbox[2]
                y2 = y1 + bbox[3]
                x_min = min(x_min,x1)
                y_min = min(y_min,y1)
                x_max = max(x_max,x2)
                y_max = max(y_max,y2)

            x_min = int(max(0,x_min-space))
            y_min = int(max(0,y_min-space))
            x_max = int(min(511,x_max+space))
            y_max = int(min(511,y_max+space))

            ls[name] = [x_min,y_min,x_max,y_max]
        with open(out_path,"w") as f:
            json.dump(ls,f)
# make_fg_bboxes(base_path="../data/",parts=["train","val","test"],space=25)
def bbox_infoes(parts , base_path):
    for part in parts:
        print(f"for {part}")
        path = os.path.join(base_path , f"{part}_fg_bboxes.json")
        with open( path, "r") as f : 
            data = json.load(f)
        bboxes = [[],[],[],[]]
        for key , bbox in data.items():
            bboxes[0] += [bbox[0]]
            bboxes[1] += [bbox[1]]
            bboxes[2] += [bbox[2]]
            bboxes[3] += [bbox[3]]

        mean_xmin , med_xmin  = np.mean(bboxes[0]) , np.median(bboxes[0])
        mean_ymin , med_ymin = np.mean(bboxes[1]) , np.median(bboxes[1])
        mean_xmax , med_xmax = np.mean(bboxes[2]) , np.median(bboxes[2])
        mean_ymax , med_ymax = np.mean(bboxes[3]) , np.median(bboxes[3])

        max_xmin , min_xmin  = np.max(bboxes[0]) , np.min(bboxes[0])
        max_ymin , min_ymin = np.max(bboxes[1]) , np.min(bboxes[1])
        max_xmax , min_xmax = np.max(bboxes[2]) , np.min(bboxes[2])
        max_ymax , min_ymax = np.max(bboxes[3]) , np.min(bboxes[3])


        print(f"mean = {mean_xmin} , {mean_ymin} , {mean_xmax} , {mean_ymax}")
        print(f"median = {med_xmin} , {med_ymin} , {med_xmax} , {med_ymax}")
        print(f"min = {min_xmin} , {min_ymin} , {min_xmax} , {min_ymax}")
        print(f"max = {max_xmin} , {max_ymin} , {max_xmax} , {max_ymax}")
        print("-"*100)
# bbox_infoes(parts=["train","val","test"],base_path="../data/")

def class_weighting(method,class_counts,**kwargs):
    if(kwargs["use_pixel_counts"]):
        print("using pixel counts")
        with open("./data/train_pixel_counts.json","r") as f:
            train_class_counts = json.load(f)
        counts = [0]*(len(train_class_counts))
        for k,v in train_class_counts.items():
            counts[int(k)] = int(v)
        counts = np.array(counts,dtype=np.float64)
    else :
        print("using class counts")
        counts = np.array(class_counts,dtype=np.float64)

    if(method=="median"):
        print("median weights being used")
        median_count = np.median(counts)
        weights = median_count/np.array(counts)
        
    elif(method=="log"):
        print("log weights being used")
        total = np.sum(counts)
        weights = np.log(total/np.array(counts))
        weights = (weights / weights.mean())
        weights[0]=0.1
    elif(method=="beta"):
        print("beta weights being used")
        b = kwargs["b"]
        weights = (1-b)/(1-np.power(b,counts))
        weights = weights / weights.sum()
        weights[12] = 0.25
    else:
        print("no class weights being used")
        return None
    return weights.tolist()

@torch.no_grad()
def freeze_binary_encoder(binary_encoder):
    
    for param in binary_encoder.parameters():
        param.requires_grad = False
    return binary_encoder