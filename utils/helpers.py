import os
import zarr
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import re
import torch
from itertools import repeat
import random
import copy
import matplotlib.pyplot as plt
import zarr
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize
from sklearn.metrics import confusion_matrix  , average_precision_score
from skimage.filters import frangi
import seaborn as sns
from torch_geometric.loader import DataLoader as GnnDataLoader
from torch_geometric.utils import subgraph
from collections import Counter
import json
import albumentations as A
from scipy.ndimage import distance_transform_edt
###IE###
from .image_inspection import potentially_problematic , bad , might_be_wrong
from .dataset import UnetExampleDataset , TrainUnetDataset ,GnnDataset
from .preprocessing import improve_contrast
###SS###

def read_images(
        paths, 
        part,
        bootstrap,
        training_mode,
        binary_mode,
        side_mode,
        in_c,
        class_count,
        old_class_map=None,
        collapse=None,
        max_workers=None,
        hard_classes = None,
        chosen_labels = None,
        cut_borders=True,
        lut=None
    ):
    
    base_path = Path(paths["base_path"])
    images_folder = paths["images_folder"]
    generated_folder = paths["generated_folder"]
    labels_folder = paths["labels_folder"]
    bootstrap_folder = paths["bootstrap_folder"]
    filters_folder = paths["filter_folder"]
    border_removed_folder = paths["border_cords_folder"]
    clahe_folder = paths["clahe_folder"]
    hard_vessel_folder = paths["hard_vessel_folder"]
    topology_folder = paths["topology_folder"]
    skel_path = paths["skel_folder"]
    ignore_labels_folder = paths["ignore_labels_folder"]
    side_folder = Path(paths["side_folder"])
    

    images_base = base_path / images_folder / part
    clahe_base = base_path / clahe_folder / part
    generated_base = base_path / generated_folder / part
    labels_base = base_path / labels_folder / part
    hard_vessel_base = base_path / hard_vessel_folder / part
    filters_base = base_path / filters_folder / part
    topology_base = base_path / topology_folder / part
    skel_base = base_path / skel_path / part
    ignore_labels_base= base_path / ignore_labels_folder
    border_removed_base = base_path / border_removed_folder / part / "cords.csv"
    bootstrap_base = base_path / bootstrap_folder 
    side_json = side_folder / f"{part}_side_labels.json"

    black_list = bad + potentially_problematic + might_be_wrong

    mask_classes = [10,11,12,14,15,17,18,19,24]
    if(bootstrap):
        bootstrap_levels = os.listdir(bootstrap_base)
        bootstrap_levels = list(map(int, bootstrap_levels))
        bootstrap_levels.sort()
        last_level = bootstrap_levels[-1]
        print(last_level)
        print(generated_base)
    else:
        last_level = 0
        print(last_level)

    # cords_df = pd.read_csv(border_removed_base)
    with open(side_json,"r") as f:
        side_dict = json.load(f)

    image_hard_count = []
    image_names = sorted([p.name for p in os.scandir(images_base) if p.is_file()])

    easy_classes = np.array([1,2,3,4,5,6,7,8,13,16,20])

    easy_map = np.zeros(26)
    easy_map[easy_classes] = 1

    def _read_one(fname):
        name_stem = Path(fname).stem
        img_path = images_base / fname
        clahe_path = clahe_base / fname
        generated_path = generated_base / f"{name_stem}.npy"
        filter_path = filters_base / f"{name_stem}.npy"
        label_path = labels_base / f"{name_stem}.npy"
        hard_idx_path = hard_vessel_base / f"{name_stem}.npy"
        skel_mask_path = skel_base / f"{name_stem}.npy"
        branch_box_path = topology_base / f"{name_stem}.txt"

        bootstrap_path = bootstrap_base /str(last_level)/part/f"{name_stem}.png"
        side = int(side_dict[name_stem])
        if(in_c==1):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        elif(in_c==2):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            clahe = cv2.imread(str(clahe_path), cv2.IMREAD_GRAYSCALE)
            img = np.stack([clahe,img],axis=-1)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if(bootstrap):
            label = cv2.imread(bootstrap_path,cv2.IMREAD_GRAYSCALE)//255
        else:
            label = np.load(str(label_path))

        w = 1
        uniques = np.unique(label)
        if(hard_classes is not None):
            for u in uniques:
                if(u in hard_classes):
                    w+=1

        skel_mask = np.load(skel_mask_path).astype(int)
        if(training_mode=="binary" or training_mode=="branch"):
            label = (label!=0).astype(int)
            skel_mask = (skel_mask!=0).astype(int)
        hard_idx = np.load(hard_idx_path)

        outs = {
            "img" : img,
            "hard_idx" : hard_idx,
            "label" : label,
            "skel_mask" : skel_mask,
            "side" : side,
            "name_stem" : name_stem,
            "w" : w
        }
        return outs

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(32, cpu * 4)

    results = []
    small_masks = None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for outs in tqdm(ex.map(_read_one, image_names), total=len(image_names)):
            side = outs["side"]
            label = outs["label"]
            skel_mask = outs["skel_mask"]
            name_stem = outs["name_stem"]
            w = outs["w"]

            if(training_mode=="multi"):
                if side_mode == "left" : 
                    if(side == 0):
                        continue
                    else:
                        hard_mask = np.zeros_like(label)
                        for u in mask_classes:
                            hard_mask[label == u] =1
                     
                        label[label!=0]-=4
                        label[label==20]-=4
                        label[label==21]-=4

                        skel_mask[skel_mask!=0]-=4
                        skel_mask[skel_mask==20]-=4
                        skel_mask[skel_mask==21]-=4

                        child_ids = [i for i, nm in old_class_map.items() if collapse[nm] == '0']
                        small_masks = np.isin(label, child_ids)
                        if(lut is not None):
                            label = lut[label]
                            skel_mask = lut[skel_mask]

                        
                elif side_mode == "right" : 
                    if(side == 1):
                        continue
                    else:
                        
                        if(class_count==6):
                            mask_16a = np.zeros_like(label)
                            mask_16a[label==21] = 1
                            label[label>=20]=5
                            skel_mask[skel_mask>=20]=5
                        elif(class_count==9):
                            label[label>=20]-=15
                            skel_mask[skel_mask>=20]-=15
                
            if(training_mode=="binary"):
                if side_mode == "left" : 
                    if(side == 0):
                        continue
                elif side_mode == "right" : 
                    if(side == 1):
                        continue
                        
            if(part == "train"):
                
                
                ignore_mask = np.load(ignore_labels_base / f"{name_stem}.npy")
                if(small_masks is not None):
                    ignore_mask |= small_masks
                # overlap_mask = np.load(f"./dataset/syntax/overlaps/train/{name_stem}.npy")
                # ignore_mask = (ignore_mask | overlap_mask).astype(np.uint8)
            else:
                ignore_mask = np.zeros_like(label)
                        # label[label>=20]=5
            # if(int(name_stem) in black_list) :
            #     continue


            outs["skel_mask"] = skel_mask
            outs["label"] = label
            outs["ignore_mask"] = ignore_mask
            image_hard_count.append(w)
            results.append(outs)


    if(hard_classes is not None):
        weights = np.array(image_hard_count)
        return results , weights , last_level
    return results

def read_gnn_dataset(paths,part,chosen_index,training_mode,family,
                     use_binary_model=False,max_workers=None,luts=None):
    
    base_path = Path(paths["base_path"])
    base_part_path = base_path / part

    names = os.listdir(base_part_path)
    def _read_one(fname,chosen_index):
        name = Path(fname).stem
        feature_path_base = base_part_path / name / "features" 
        edge_path_base = base_part_path / name / "edges" 
        label_path_base = base_part_path / name / "labels" 
        pred_mask_path_base = base_part_path / name / "pred_masks" 
        true_mask_path_base = base_part_path / name / "true_masks"
        weight_path_base = base_part_path / name / "node_weights"

        pred_families = os.listdir(feature_path_base)
        gt_families = os.listdir(true_mask_path_base)
        pred_families = [Path(c).stem for c in pred_families if Path(c).stem in family]
        gt_families = [Path(c).stem for c in gt_families if Path(c).stem in family]


        outs = {
            "correct" : {
                "features":{},
                "edges":{},
                "labels":{},
                "preds":{},
                "gts":{},
                "weights" : {}
            },
            "missed":{
                "gts":{}
            },
            "extra":{
                "features":{},
                "edges":{},
                "labels":{},
                "preds":{},
                "weights" : {}
            }
        }
    
        model_missed_classes = np.setdiff1d(gt_families, pred_families)
        model_fp_classes = np.setdiff1d(pred_families,gt_families)
        model_correct_classes = np.intersect1d(gt_families, pred_families)

        for c in model_correct_classes:
            f = np.load(feature_path_base / f"{c}.npy")
    
            outs["correct"]["features"][c] = f[...,chosen_index]
            outs["correct"]["edges"][c] = np.load(edge_path_base / f"{c}.npy")
            outs["correct"]["labels"][c] = np.load(label_path_base / f"{c}.npy")
            outs["correct"]["preds"][c] = cv2.imread(pred_mask_path_base / f"{c}.png",cv2.IMREAD_GRAYSCALE)//255
            outs["correct"]["gts"][c] = np.load(true_mask_path_base / f"{c}.npy")
            outs["correct"]["weights"][c] = np.load(weight_path_base / f"{c}.npy")
            if(luts[c] is not None):
                outs["correct"]["labels"][c] = luts[c][outs["correct"]["labels"][c]]
        for c in model_missed_classes:
            outs["missed"]["gts"][c] = np.load(true_mask_path_base / f"{c}.npy")
    
        for c in model_fp_classes:
            f = np.load(feature_path_base / f"{c}.npy")

            outs["extra"]["features"][c] = f[...,chosen_index]
            outs["extra"]["edges"][c] = np.load(edge_path_base / f"{c}.npy")
            outs["extra"]["labels"][c] = np.load(label_path_base / f"{c}.npy")
            outs["extra"]["preds"][c] = cv2.imread(pred_mask_path_base / f"{c}.png",cv2.IMREAD_GRAYSCALE)//255
            outs["extra"]["weights"][c] = np.load(weight_path_base / f"{c}.npy")
            if(luts[c] is not None):
                outs["correct"]["labels"][c] = luts[c][outs["correct"]["labels"][c]]

        outs["name"] = name
        return outs

    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(32, cpu * 4)

    results = []
    co=0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for outs in tqdm(ex.map(_read_one, names,repeat(chosen_index)), total=len(names)):
            co+=1

            results.append(outs)
    print(co)
    return results

def mimority_pr_auc(y_trues,y_pred_probs):
    y_trues_minority = 1 - np.array(y_trues)
    y_pred_probs_minority = 1 - np.array(y_pred_probs)
    auc_minority = average_precision_score(y_trues_minority, y_pred_probs_minority)

    return auc_minority
@torch.no_grad()
def generate_images(model,train_images,valid_images,normalizer,transforms,args,out_folder):

    model.eval()
    train_ds = UnetExampleDataset(
        normalizer = normalizer,
        data = train_images,
        transform=transforms
    )
    valid_ds = UnetExampleDataset(
        normalizer = normalizer,
        data = valid_images,
        transform=transforms
    )
    train_loader = DataLoader(
        train_ds,
        batch_size = 1  ,
        num_workers = 0 ,
        pin_memory=True,
        shuffle=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size = 1  ,
        num_workers = 0 ,
        pin_memory=True,
        shuffle=False,
    )
    train_out = os.path.join(out_folder,"train")
    valid_out = os.path.join(out_folder,"val")
    # out = next(iter(valid_loader))

    for imgs,names in tqdm(train_loader) : 
        imgs = imgs.to(args["device"])
        with torch.autocast(device_type=args["device"],dtype=torch.float16):
            mask_preds = model(imgs)[0]
        masks = torch.argmax(mask_preds,dim=1)
        mask = masks[0].cpu().numpy().astype(np.uint8)*255
        cv2.imwrite(f"{train_out}/{names[0]}.png",mask)

    for imgs,names in tqdm(valid_loader) : 
        imgs = imgs.to(args["device"])
        with torch.autocast(device_type=args["device"],dtype=torch.float16):
            mask_preds = model(imgs)[0]
        masks = torch.argmax(mask_preds,dim=1)

        mask = masks[0].cpu().numpy().astype(np.uint8)*255
        cv2.imwrite(f"{valid_out}/{names[0]}.png",mask)

@torch.no_grad()
def generate_all_masks(model,train_loader,valid_loader,test_loader,output_base="./dataset/syntax/"):
    output_base = os.path.join(output_base,"generated_mask_probs")
    train_base = os.path.join(output_base,"train")
    valid_base = os.path.join(output_base,"val")
    test_base = os.path.join(output_base,"test")

    os.makedirs(output_base,exist_ok=True)
    os.makedirs(train_base,exist_ok=True)
    os.makedirs(valid_base,exist_ok=True)
    os.makedirs(test_base,exist_ok=True)

    model.eval()

    for img_base , img , mask ,side, name in tqdm(train_loader):
        img = img.to("cuda")
        name = name[0]
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            pred_mask = model(img)[0]
            pred_mask = F.softmax(pred_mask,dim=1)
            pred_mask = pred_mask[0,1].cpu().numpy()
        np.save(os.path.join(train_base,f"{name}.npy"),pred_mask)

    for img_base , img , mask ,side, name in tqdm(valid_loader):
        mask = (mask!=0).int()
        img = img.to("cuda")
        name = name[0]
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            pred_mask = model(img)[0]
            pred_mask = F.softmax(pred_mask,dim=1)
            pred_mask = pred_mask[0,1].cpu().numpy()
            print(pred_mask.max(),pred_mask.min(),pred_mask.mean())
        np.save(os.path.join(valid_base,f"{name}.npy"),pred_mask)
        
    for img_base , img , mask ,side, name in tqdm(test_loader):
        mask = (mask!=0).int()
        img = img.to("cuda")
        name = name[0]
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            pred_mask = model(img)[0]
            pred_mask = F.softmax(pred_mask,dim=1)
            pred_mask = pred_mask[0,1].cpu().numpy()
        np.save(os.path.join(test_base,f"{name}.npy"),pred_mask)

@torch.no_grad()
def check_error_for_imgs(model,train_images,valid_images,normalizer,
                         transforms,loss_fn,args):
    model.eval()
    train_ds = UnetExampleDataset(
        normalizer = normalizer,
        data = train_images,
        transform=transforms,
        loss_check=True
    )
    valid_ds = UnetExampleDataset(
        normalizer = normalizer,
        data = valid_images,
        transform=transforms,
        loss_check=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size = 1  ,
        num_workers = 0 ,
        pin_memory=True,
        shuffle=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size = 1  ,
        num_workers = 0 ,
        pin_memory=True,
        shuffle=False,
    )
    loss_history = {"names":[],"total_loss":[]}
    image_history = {}
    for raw_img , imgs,gt_masks,names in tqdm(valid_loader) : 
        imgs = imgs.to(args["device"])
        gt_masks = gt_masks.to(args["device"])
        with torch.autocast(device_type=args["device"],dtype=torch.float16):
            pred_masks = model(imgs)
            loss , loss_dict = loss_fn(
                pred_masks = pred_masks[0],
                gt_masks = gt_masks
            )
        for key,l in loss_dict.items():
            if(key not in loss_history):
                loss_history[key]=[]
            loss_history[key].append(l.cpu().item())
        loss_history["names"].append(names[0])
        loss_history["total_loss"].append(
            loss.cpu().item()
        )
        

        image_history[names[0]]=[
            gt_masks[0,:,:].cpu().numpy(),
            raw_img[0,0,:,:].cpu().numpy()
        ]
    return loss_history , image_history
 
 
def effective_num_weights(class_counts, knee=100.0, beta=None, normalize=True):
    n = np.asarray(class_counts, dtype=np.float64)
    present = n > 0

    m = n / n[present].min()
    if(knee is None):
        knee = np.max(n[present])/np.min(n[present])
        print(f"knee is set to : {knee}")
    if beta is None:
        if not (knee > 1.0):
            raise ValueError("knee must be > 1")
        beta = 1.0 - 1.0 / knee
        print(f"beta is set to : {beta}")

    w = np.zeros_like(n)
    if beta == 0.0:
        w[present] = 1.0
    else:

        denom = -np.expm1(m[present] * np.log(beta))
        w[present] = (1.0 - beta) / denom
 
    if normalize:
        w = w / w[present].mean()
 
    return w


def count_class_pixels(train_images, class_count):
    class_counts = np.zeros(class_count, dtype=np.int64)
 
    for data in train_images:
        label = data["label"]
        flat = np.asarray(label).reshape(-1)
        if flat.size == 0:
            continue
        class_counts += np.bincount(flat, minlength=class_count)[:class_count]
 
    return class_counts


def _sobel_kernels(device, dtype):

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky

def improved_vessel_map(image, sigma=1.0):

    padding = int(3 * sigma)

    x = torch.arange(-padding, padding + 1, device=image.device).float()
    gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
    kernel = (gauss.view(1, 1, -1, 1) * gauss.view(1, 1, 1, -1))
    kernel /= kernel.sum()
    
    smoothed = F.conv2d(image, kernel, padding=padding)
    

    kx, ky = _sobel_kernels(image.device, image.dtype)
    gx = F.conv2d(smoothed, kx, padding=1)
    gy = F.conv2d(smoothed, ky, padding=1)
    
    edge = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
    
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-6)
    return edge

@torch.no_grad()
def popov_edge_map(gt_mask, eps = 1e-6) :
    """
    input : gt_mask (B,1,H,W)
    output : edge_norm (B,1,H,W)
    """
    if gt_mask.ndim != 4 or gt_mask.size(1) != 1:
        raise ValueError("gt_mask must be (B,1,H,W)")

    gt = gt_mask.float()
    device, dtype = gt.device, gt.dtype
    kx, ky = _sobel_kernels(device, dtype)

    gx1 = F.conv2d(gt, kx, padding=1)
    gy1 = F.conv2d(gt, ky, padding=1)
    edge1 = gx1.pow(2) + gy1.pow(2) 

    gx2 = F.conv2d(edge1, kx, padding=1)
    gy2 = F.conv2d(edge1, ky, padding=1)
    edge2 = torch.sqrt(gx2.pow(2) + gy2.pow(2) + eps)

    b = edge2.flatten(1)
    mn = b.min(dim=1).values.view(-1, 1, 1, 1)
    mx = b.max(dim=1).values.view(-1, 1, 1, 1)
    edge_norm = (edge2 - mn) / (mx - mn + eps)
    return edge_norm

@torch.no_grad()
def make_example_datasets(valid_images,image_names,transform,normalizer):
    img_dict = {"easy":[],"normal":[],"hard":[],"very hard":[]}
    for data  in valid_images : 
        name_stem = data["name_stem"]
        for diff in image_names:
            if(name_stem in image_names[diff]):
                img_dict[diff].append(data)
                break
    for diff , images in img_dict.items():
        ds = TrainUnetDataset(
            transform = transform,
            data = images,
            example_phase=True,
            normalizer = normalizer
        )
        dl = DataLoader(
            ds,
            batch_size=1,
            num_workers=0,
            shuffle=False,
        )
        img_dict[diff] = dl

    return img_dict



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
def TP_TN_FP_FN(preds,gt,process_preds=False,return_TN=False):
    if(process_preds):
        preds_argmax = torch.argmax(preds,dim=1)
        onehot_preds = F.one_hot(preds_argmax,num_classes=preds.shape[1])
        pred_onehot = onehot_preds.permute(0, 3, 1, 2).float()
    else :
        pred_onehot = preds
        
    onehot_gt = F.one_hot(gt,num_classes=pred_onehot.shape[1])
    onehot_gt = onehot_gt.permute(0, 3, 1, 2).float()
    TN = 0
    if(return_TN):
        TN = (((1-onehot_gt)*(1-pred_onehot)).sum(dim=(0,2,3))).cpu()

    TP = ((onehot_gt*pred_onehot).sum(dim=(0,2,3))).cpu()
    FP = (((1-onehot_gt)*pred_onehot).sum(dim=(0,2,3))).cpu()
    FN = ((onehot_gt*(1-pred_onehot)).sum(dim=(0,2,3))).cpu()
    return TP , TN , FP , FN

def draw_mask(image,mask,training_mode,args=None,colors=None):
    if(training_mode=="cleaning"):
    # print(training_mode)
        return mask
    img = image.copy().astype(np.uint8)
    m = mask.astype(np.int64)
    if(colors is None):
        colors = np.array([(0,255,0)]*25,dtype=np.uint8)
    img[m>0] = colors[m[m>0]-1]
    return img



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
def compute_confution_matrix(data_loader,model,class_maps,
                             output_folder_path=None,
                             draw_plot = True,class_count=26,
                             training_mode = "binary"
                             ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf_mat = torch.zeros((class_count,class_count))
    model.eval()
    for data in tqdm(data_loader):
        imgs = data["imgs"].to(device)
        gt_masks = data["mask"].to(device)
        sides = data["side"].int().to(device).reshape(-1)
        with torch.autocast(device_type=device,dtype=torch.float16):
            
            pred_masks,aux_loss  = model(imgs,ep=100000000,gt_mask=gt_masks)
            pred_masks = pred_masks[0]
           
            # pred_targets : B x k x H x W
            # targets : B x H x W

        gt_masks = gt_masks.reshape(-1)
        pred_masks = torch.argmax(pred_masks,dim=1).view(-1) # B x H x W
        encoded_results = (gt_masks*class_count + pred_masks).cpu() # # B x H x W
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

@torch.no_grad()
def compute_gnn_confution_matrix(models_label_predictions,class_maps,
                                 output_folder_path,family):
    
    class_map = class_maps[family]

    classnames = [0]*len(class_map)
    for i,name in class_map.items():
        classnames[i]=name
        
    for part , results in models_label_predictions.items():
        
        all_labels = results[family]["true"]
        all_preds = results[family]["pred"]

        plt.figure(figsize=(15,10))
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax = sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            xticklabels=classnames,
            yticklabels=classnames,
            cmap="Blues"
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{part} Confusion Matrix {family}")
        plt.tight_layout()
        if(output_folder_path):
            out_path = os.path.join(output_folder_path,f"{part}_conf_mat_{family}.png")
            plt.savefig(out_path)

def denormalize(img_norm,channels):
    if(channels==3):
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img_norm * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img, 0.0, 1.0)
        img_u8 = (img * 255).astype(np.uint8)
        return img_u8
    

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


import segmentation_models_pytorch as smp


def load_model(model_class,path,model_inputs,device,return_args = False,task_type="cnn",model_name=None):
    # best_model = torch.load(f"{path}/model.pth",weights_only=False)
    if(model_name == None):
        model_name = "model"
    pth = torch.load(f"{path}/{model_name}.pth",weights_only=False)
    args_path = os.path.join(path,"args.json")
    with open(args_path,"r") as f:
        args = json.load(f)

    if(task_type !="gnn"):
        train_transforms = A.load(os.path.join(path,"train_augs.json"))
    else:
        train_transforms = None
    model_inputs["args"]=args
    best_model = model_class(**model_inputs)
    best_model.load_state_dict(pth)
    best_model = best_model.to(device)
    best_model.eval()
    if(return_args):
        return best_model,args,train_transforms
    return best_model

def make_result_dataloaders(
        train_images,valid_images,
        normalizer,transforms,test_images=None):
    train_ds = UnetExampleDataset(
        transform=transforms,
        data=train_images,
        normalizer=normalizer,
        gnn_ds=True
    )
    train_loader = DataLoader(train_ds,batch_size=1,shuffle=False)

    valid_ds = UnetExampleDataset(
        transform=transforms,
        data=valid_images,
        normalizer=normalizer,
        gnn_ds=True
    )
    valid_loader = DataLoader(valid_ds,batch_size=1,shuffle=False)

    if(test_images is not None):
        test_ds = UnetExampleDataset(
            transform=transforms,
            data=test_images,
            normalizer=normalizer,
            gnn_ds=True
        )
        test_loader = DataLoader(test_ds,batch_size=1,shuffle=False)

        return train_loader , valid_loader , test_loader
    return train_loader , valid_loader 

def seed_worker(worker_id):
    ws = torch.initial_seed() % 2**32        # torch redraws this every epoch
    np.random.seed(ws); random.seed(ws)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset.transform, "set_random_seed"):
        info.dataset.transform.set_random_seed(int(ws))   # <- the actual fix


def make_training_dataloaders(
        train_images,valid_images,
        normalizer,train_transforms,
        test_transforms,args,test_images=None,sampler=None):
    train_ds = TrainUnetDataset(
        transform = train_transforms,
        data = train_images , 
        normalizer = normalizer,
        example_phase=False,
        part = "train"
    )
    valid_ds = TrainUnetDataset(
        transform = test_transforms,
        data = valid_images,
        normalizer = normalizer,
        example_phase=False,
        part = "val"
    )

    if(sampler is not None):

        train_loader = DataLoader(
            train_ds,
            batch_size = args["batch_size"] ,
            num_workers = args["num_workers"] ,
            pin_memory=True,
            persistent_workers= True if args["num_workers"] else False ,
            sampler=sampler,
            drop_last=True,
            worker_init_fn=seed_worker
        )
        print("sampler is set for trainloader")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size = args["batch_size"] ,
            num_workers = args["num_workers"] ,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True if args["num_workers"] else False,
            drop_last=True,
            worker_init_fn=seed_worker
        )
    valid_loader = DataLoader(
        valid_ds,
        batch_size = 1,
        pin_memory=True,
        shuffle=False,

    )
    if(test_images is not None):

        test_ds = TrainUnetDataset(
            transform = test_transforms,
            data = test_images,
            normalizer = normalizer,
            example_phase = False,
            part = "test"
        )
        test_loader = DataLoader(
            test_ds,
            batch_size = 1  ,
            pin_memory=True,
            shuffle=False,
        )
        return train_loader , valid_loader , test_loader
    return train_loader , valid_loader

def make_gnn_exmaple_loaders(args,train_images,valid_images,
                             Example_DL,family,test_images=None):
    train_ds = GnnDataset(
        data = train_images , 
        part = "train",
        eval_mode = True,
        training_mode = args["training_mode"],
        family=family
    )
    valid_ds = GnnDataset(
        data = valid_images , 
        part = "val",
        training_mode = args["training_mode"],
        eval_mode= True,
        family=family
    )

    train_loader = Example_DL(
        train_ds,
        batch_size = 1 ,
        num_workers = 0 ,
        pin_memory=True,
        shuffle=True
    )
    valid_loader = Example_DL(
        valid_ds,
        batch_size = 1 ,
        pin_memory=True,
        shuffle=False,
    )
    example_trainloaders = {
        "train":train_loader,
        "val":valid_loader
    }
    if(test_images is not None):
        test_ds = GnnDataset(
            data = test_images , 
            part = "test",
            selected_features = args["selected_features"],
            eval_mode = True,
            family=family
        )
        test_loader = Example_DL(
            test_ds,
            batch_size = 1  ,
            pin_memory=True,
            shuffle=False,
        )
        example_trainloaders["test"] = test_loader
    return example_trainloaders


def draw_corl_matrix(train_images,feature_keys,family,plot_shape = (1,1)):
    all_features = {key:[] for key in family}
    all_labels = {key:[] for key in family}

    for data in tqdm(train_images):
        correct = data["correct"]
        if(len(correct["features"]) == 0):
            continue
        for c,f in correct["features"].items():
            all_features[c].append(f)
        for c,l in correct["labels"].items():
            all_labels[c].append(l)

    
    for c in family:
        all_features[c] = np.concatenate(all_features[c],axis=0)
        all_labels[c] = np.concatenate(all_labels[c],axis=0)
        print(f"{c} min max features : ",all_features[c].min(),all_features[c].max())
    

    multi_df = {key:{} for key in family}
    counts = {key:{} for key in family}
    fig, axes = plt.subplots(plot_shape[0], plot_shape[1],
                         figsize=(10 * plot_shape[1], 8 * plot_shape[0]))
    axes = np.atleast_1d(axes).flatten() 
    for i,c in enumerate(multi_df) :
        for j in range(len(feature_keys)):
            key = feature_keys[j] 
            multi_df[c][key] = all_features[c][...,j]
        multi_df[c]["class"] = all_labels[c]

        multi_df[c] = pd.DataFrame(multi_df[c]).copy()
        multi_corr_matrix = multi_df[c].corr(method="pearson")
        plt.subplot(plot_shape[0],plot_shape[1],i+1)
        
        sns.heatmap(multi_corr_matrix, 
                    annot=True, 
                    cmap='coolwarm',
                    mask=np.tri(len(multi_df[c].columns)).T, 
                    center=0,
                    ax=axes[i])
        axes[i].set_title(f'Correlation: {c}')

        co = Counter(all_labels[c].tolist())
        print(f"{c} : {co}")
        counts[c] = co

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.show()
    return counts

def create_skeleton_bridge_mask(expert_mask, pred_mask, new_vessel_width=3):

    expert_bool = expert_mask > 0
    pred_bool = pred_mask > 0

    pred_skeleton = skeletonize(pred_bool)

    new_branches_skel = np.logical_and(pred_skeleton, np.logical_not(expert_bool))

    kernel = np.ones((new_vessel_width, new_vessel_width), np.uint8)
    new_vessels_dilated = cv2.dilate(new_branches_skel.astype(np.uint8), kernel)

    combined = np.logical_or(expert_bool, new_vessels_dilated > 0)
    
    return (combined * 255).astype(np.uint8)

def _bootstrap_once(
        pred_mask,
        binary_mask,
        img,
        part_path,
        name,
        gt_mask_mode,
        kernel,
        new_vessel_width,
        iterations,
        comp,
        use_erode):

        if(gt_mask_mode == "real_pred"):
            if(comp==True):
                full_plot = np.zeros((1048,1048,3))
                thin_mask1 = cv2.erode(pred_mask, kernel, iterations=1)
                thin_mask2 = cv2.erode(pred_mask, kernel, iterations=2)
                thin_mask3 = cv2.erode(pred_mask, kernel, iterations=3)

                new_mask1 = (np.logical_or(thin_mask1, binary_mask) * 255).astype(np.uint8)
                new_mask2 = (np.logical_or(thin_mask2, binary_mask) * 255).astype(np.uint8)
                new_mask3 = (np.logical_or(thin_mask3, binary_mask) * 255).astype(np.uint8)
                
                img0__gt = img.copy()
                img0__pred = img.copy()

                img1__gt = img.copy()
                img1__pred = img.copy()

                img2__gt = img.copy()
                img2__pred = img.copy()

                img3__gt = img.copy()
                img3__pred = img.copy()

                img0__gt[binary_mask!=0] = [255,0,0]
                img0__pred[pred_mask!=0] = [0,0,255]

                img1__gt[thin_mask1!=0] = [0,0,255]
                img1__gt[binary_mask!=0] = [255,0,0]
                img1__pred[new_mask1!=0] = [0,0,255]

                img2__gt[thin_mask2!=0] = [0,0,255]
                img2__gt[binary_mask!=0] = [255,0,0]
                img2__pred[new_mask2!=0] = [0,0,255]

                img3__gt[thin_mask3!=0] = [0,0,255]
                img3__gt[binary_mask!=0] = [255,0,0]
                img3__pred[new_mask3!=0] = [0,0,255]

                

                full_plot = draw_fast_plots([
                    img0__pred,img0__gt,
                    img1__pred,img1__gt,
                    img2__pred,img2__gt,
                    img3__pred,img3__gt
                ],plot_shape=(4,2))

                new_mask = full_plot
            else:
                if(use_erode):
                    p_mask = cv2.erode(pred_mask, kernel, iterations=iterations)
                else:
                    p_mask = pred_mask
                new_mask = (np.logical_or(p_mask, binary_mask) * 255).astype(np.uint8)
        elif(gt_mask_mode =="pred_skel"):
            if(comp==True):
                new_mask1 = create_skeleton_bridge_mask(binary_mask.copy(),pred_mask.copy(),
                                                new_vessel_width=2)
                new_mask2 = create_skeleton_bridge_mask(binary_mask.copy(),pred_mask.copy(),
                                                new_vessel_width=3)
                new_mask3 = create_skeleton_bridge_mask(binary_mask.copy(),pred_mask.copy(),
                                                new_vessel_width=4)
                new_mask4 = create_skeleton_bridge_mask(binary_mask.copy(),pred_mask.copy(),
                                                new_vessel_width=5)
                new_mask5 = create_skeleton_bridge_mask(binary_mask.copy(),pred_mask.copy(),
                                                new_vessel_width=6)
                new_mask6 = create_skeleton_bridge_mask(binary_mask.copy(),pred_mask.copy(),
                                                new_vessel_width=7)
                
                img_gt = img.copy()
                img_pred = img.copy()
                img1 = img.copy()
                img2 = img.copy()
                img3 = img.copy()
                img4 = img.copy()
                img5 = img.copy()
                img6 = img.copy()
                
                img_gt[binary_mask!=0] = [255,0,0]
                img_pred[pred_mask!=0] = [0,0,255]
                img1[new_mask1!=0] = [0,0,255]
                img2[new_mask2!=0] = [0,0,255]
                img3[new_mask3!=0] = [0,0,255]
                img4[new_mask4!=0] = [0,0,255]
                img5[new_mask5!=0] = [0,0,255]
                img6[new_mask6!=0] = [0,0,255]
                full_plot = draw_fast_plots([
                    img,img_gt,img_pred,
                    img1,img2,img3,
                    img4,img5,img6,
                ],plot_shape=(3,3))
                new_mask = full_plot
            else:
                new_mask = create_skeleton_bridge_mask(binary_mask,pred_mask,
                                                new_vessel_width=new_vessel_width)
            
        boot_path = os.path.join(
            part_path,
            f"{name}.png"
        )
        cv2.imwrite(boot_path,new_mask)
        return name

def bootstrap_model(model,last_boot_level,args,dataloaders,gt_mask_mode,new_vessel_width,
                    model_name,max_workers=4,max_pending=8,iterations=1,
                    comp = False,use_erode=False):
    new_boot_level = str(last_boot_level + 1)
    
    boot_base_path = os.path.join(
        args["paths"]["base_path"],
        args["paths"]["bootstrap_folder"],
        new_boot_level
    )
    
    os.makedirs(boot_base_path,exist_ok=True)
    for file in Path(boot_base_path).glob("*.txt"):
        file.unlink()

    Path(f"{boot_base_path}/{model_name}.txt").touch(exist_ok=True)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    model.eval()

    pending = set()

    
    
    for part , dataloader in dataloaders.items():
        part_path = os.path.join(boot_base_path,part)
        os.makedirs(part_path,exist_ok=True)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(dataloader, desc=f"{part}")
            for raw_imgs,imgs, mask, side,names in pbar:
                binary_mask = (mask>=0.5).int()
                imgs = imgs.to(args["device"])
                name = names[0]
                with torch.autocast(device_type=args["device"],dtype=torch.float16):
                    pred_logits = model(imgs)[0]
                
                pred_mask = torch.argmax(pred_logits,dim=1)
                pred_mask = pred_mask.cpu().numpy()
                binary_mask = binary_mask.numpy()
                job = (
                    pred_mask[0].astype(np.uint8),
                    binary_mask[0].astype(np.uint8),
                    raw_imgs[0].permute(1,2,0).numpy(),
                    part_path,
                    name,
                    gt_mask_mode,
                    kernel,
                    new_vessel_width,
                    iterations,
                    comp,
                    use_erode
                )
                pending.add(executor.submit(_bootstrap_once, *job))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()

            # finish remaining jobs
            for f in tqdm(pending, desc=f"finishing {part}"):
                f.result()


def make_skel_from_features(features,pred_mask,pred_labels,colors=None,use_color=False):
    fill_val = 0 if use_color else -1
    skel_shape = pred_mask.shape+(3,) if use_color else pred_mask.shape

    skel_mask = np.full(skel_shape,fill_val)
    i=0
    max_dim = max(pred_mask.shape[0]-1,pred_mask.shape[1]-1)
    for feature in features : 
        if(use_color):
            c = colors[pred_labels[i]]
        else:
            c = pred_labels[i]
        x,y = feature[:2]
        x =int(x*max_dim)
        y =int(y*max_dim)
        
        skel_mask[x,y]=c
        i+=1
    
    if(use_color):
        skel_mask = skel_mask.astype(np.uint8)
    return skel_mask

def reconstruct_binary_vessels(binary_mask,node_coords, node_labels):
    seed_grid = np.full(binary_mask.shape, -1, dtype=int)
    max_dim = max(binary_mask.shape[0]-1,binary_mask.shape[1]-1)

    for (y, x), label in zip(node_coords, node_labels):
        x = int(x*max_dim)
        y = int(y*max_dim)
        seed_grid[y, x] = label

    is_empty = (seed_grid == -1)
    
    distances, indices = distance_transform_edt(is_empty, return_indices=True)
    
    closest_y, closest_x = indices[0], indices[1]
    
    dense_labels = seed_grid[closest_y, closest_x]

    final_mask = np.zeros_like(binary_mask, dtype=int)
    vessel_pixels = (binary_mask > 0)
    final_mask[vessel_pixels] = dense_labels[vessel_pixels]
    
    return final_mask

def reconstruct_multi_vessels(vessel_mask,skeleton_colored,node_coords,
                               node_labels,remove_color=False):
    # making the skel mask 
    skeleton_class_map = np.full(vessel_mask.shape, -1, dtype=int)
    max_dim = max(vessel_mask.shape[0]-1,vessel_mask.shape[1]-1)
    if(remove_color):
        for (y, x), label in zip(node_coords, node_labels):
            x = int(x*max_dim)
            y = int(y*max_dim)
            skeleton_class_map[y, x] = label
        skeleton_mask = (skeleton_colored.sum(-1)!=0).astype(int)
    else:
    
        skeleton_class_map = skeleton_colored.copy()
        skeleton_mask = (skeleton_colored!=-1).astype(int)
    

    vessel_mask = vessel_mask.astype(bool)
    skeleton_mask = skeleton_mask.astype(bool)

    
    seed_mask = vessel_mask & skeleton_mask & (skeleton_class_map != -1)

    full_class_map = np.full(vessel_mask.shape, -1, dtype=np.int32)

    _, indices = distance_transform_edt(~seed_mask, return_indices=True)
    nearest_y, nearest_x = indices

    full_class_map[vessel_mask] = skeleton_class_map[
        nearest_y[vessel_mask],
        nearest_x[vessel_mask]
    ]
    return full_class_map

@torch.no_grad()
def save_cleaner_results(model,part_data,paths,th):
    model.eval()

    os.makedirs(paths["binary_result_folder"],exist_ok=True)
    os.makedirs(paths["multi_base_path"],exist_ok=True)
    os.makedirs(os.path.join(paths["multi_base_path"],paths["model_preds_folder"]),
                    exist_ok=True)
    
    for part , data in part_data.items():
        base_path = os.path.join(
            paths["binary_result_folder"],part
        )
        multi_base_path = os.path.join(
            paths["multi_base_path"],part
        )
        os.makedirs(base_path,exist_ok=True)
        os.makedirs(multi_base_path,exist_ok=True)
        os.makedirs(
            os.path.join(
                paths["multi_base_path"],
                paths["model_preds_folder"],
                part
            ),
            exist_ok=True
        )

        os.makedirs(os.path.join(multi_base_path,"features"),exist_ok=True)
        os.makedirs(os.path.join(multi_base_path,"edges"),exist_ok=True)
        os.makedirs(os.path.join(multi_base_path,"labels"),exist_ok=True)
        for features , edge_index , labels,name ,img in tqdm(data):
            full_features = np.load(
                os.path.join(
                    paths["base_path"],
                    f"{part}/features",
                    name+".npy"
                )
            )

            features = torch.from_numpy(features).float().to("cuda")
            edge_index = torch.from_numpy(edge_index).int().to("cuda")
            preds_path = os.path.join(
                base_path,name+".npy"
            )

            cnn_pred_mask = cv2.imread(os.path.join(
                paths["base_path"],
                paths["model_preds_folder"],
                part,
                name+".png"
            ),cv2.IMREAD_GRAYSCALE)

            with torch.autocast(device_type="cuda",dtype=torch.float16):
                logits = model(features,edge_index)
                probs = F.softmax(logits,dim=1)[...,1]
                probs = probs.cpu().numpy()
                pred_labels = (probs>th).astype(int)

            np.save(preds_path,probs)

            keep_mask = probs > th
            keep_idx = np.where(keep_mask)[0]
            keep_idx = torch.from_numpy(keep_idx).long()

            edge_index = edge_index.cpu()
            features = features.cpu()

            edge_index, _ = subgraph(
                subset=keep_idx,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=features.shape[0]
            )

            edge_index = edge_index.numpy()
            features = features.numpy()
            keep_idx = keep_idx.numpy()
            node_cords = full_features[...,0:2].copy()

            full_features = full_features[keep_idx]
            features = features[keep_idx]
            labels = labels[keep_idx] 
            
            feature_path = os.path.join(
                multi_base_path,"features",name+".npy"
            )
            edge_path = os.path.join(
                multi_base_path,"edges",name+".npy"
            )
            label_path = os.path.join(
                multi_base_path,"labels",name+".npy"
            )
            np.save(feature_path,full_features)
            np.save(label_path,labels)
            np.save(edge_path,edge_index)

            
            new_mask_pred = (reconstruct_binary_vessels(
                binary_mask = cnn_pred_mask//255,
                node_coords = node_cords, 
                node_labels = pred_labels
            )*255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(
                    paths["multi_base_path"],
                    paths["model_preds_folder"],
                    part,
                    name+".png"
                ),
                new_mask_pred
            )

def calaculate_gnn_multi_dice(pred_skel_mask,pred_mask,gt_mask,labels,node_coords,class_count):

    full_class_map = reconstruct_multi_vessels(
        vessel_mask = pred_mask//255,
        skeleton_colored = pred_skel_mask,
        node_coords = node_coords,
        node_labels = labels,
        remove_color = False
    )
    full_class_map[full_class_map==-1]=0
    final_pred_mask = torch.from_numpy(full_class_map).long()
    gt_mask = torch.from_numpy(gt_mask).long()

    onehot_preds = F.one_hot(
        final_pred_mask.clone().unsqueeze(0),
        num_classes=class_count
    )
    onehot_preds = onehot_preds.permute(0, 3, 1, 2).float()

    multi_TP , _ , multi_FP , multi_FN = TP_TN_FP_FN(
        onehot_preds.to("cuda"),
        gt_mask.unsqueeze(0).to("cuda"),
        process_preds=False
    )
    return multi_TP.cpu(),multi_FP.cpu(),multi_FN.cpu(),full_class_map

def calaculate_gnn_binary_dice(binary_mask,binary_gt_mask,labels,node_coords):

    full_class_map = reconstruct_binary_vessels(
        binary_mask = binary_mask//255,
        node_coords = node_coords,
        node_labels = labels,
    )

    final_pred_mask = torch.from_numpy(full_class_map).long()
    binary_gt_mask = torch.from_numpy(binary_gt_mask).long()

    # print(torch.unique(final_pred_mask))
    onehot_preds = F.one_hot(
        final_pred_mask.clone().unsqueeze(0),
        num_classes=2
    )
    onehot_preds = onehot_preds.permute(0, 3, 1, 2).float()

    binary_TP , _ , binary_FP , binary_FN = TP_TN_FP_FN(
        onehot_preds.to("cuda"),
        binary_gt_mask.unsqueeze(0).to("cuda"),
        process_preds=False
    )
    return binary_TP.cpu() , binary_FP.cpu() , binary_FN.cpu() , full_class_map
def draw_fast_plots(imgs,plot_shape,border_color = 0,border_size =20):
    if(type(imgs) == np.ndarray):
        img_list = [imgs]
    if(type(imgs) == list):
        img_list = imgs
    
    shape = img_list[0].shape
    if(len(shape)==3):
        H,W,C = shape
        full_H = H*plot_shape[0] + (border_size*(plot_shape[0]+1))
        full_W = W*plot_shape[1] + (border_size*(plot_shape[1]+1))
        full_plot = np.full((full_H,full_W,C),border_color)
    else:
        H,W = shape
        full_H = H*plot_shape[0] + (border_size*plot_shape[0])
        full_W = W*plot_shape[1] + (border_size*plot_shape[1])
        full_plot = np.full((full_H,full_W),border_color)

    i=0
    for row in range(plot_shape[0]):
        current_row = border_size*(row+1) + (img_list[0].shape[0]*row)
        current_col = border_size
        for col in range(plot_shape[1]):
            if(i==len(img_list)):
                break
            img = img_list[i]
            h = img.shape[0]
            w = img.shape[1]
            full_plot[current_row:current_row+h,current_col:current_col+w] = img
            current_col += (w +border_size)
            i+=1
    return full_plot

def ensemble_pred(models,model_inputs,device):
    pred_sums = 0
    with torch.autocast(device_type=device,dtype=torch.float16):
        for model in models : 
            pred_logits = model(*model_inputs)
            model_probs = F.softmax(pred_logits,dim=1)
            pred_sums += model_probs
    return pred_sums / len(models)

def resize_dsv(mask,size,mode):
    if(mask!=None):
        dsv_masks = F.interpolate(
            input=mask.unsqueeze(1).float(),
            size=size,
            mode=mode
        ).squeeze(1).long()
    else:
        dsv_masks = None 

    return dsv_masks


def left_mapping_function(old_class_map,new_class_map,collapse):
    if(len(new_class_map)!=17):
        final_label = {idx: collapse.get(lbl, lbl)  for idx, lbl in old_class_map.items()}
        surviving = sorted({v for v in final_label.values() if v != '0'}, key=int)
        new_class_map = {i + 1: lbl for i, lbl in enumerate(surviving)}
        label_to_new = {lbl: i for i, lbl in new_class_map.items()}
        label_to_new['0'] = 0   

        remap = {0: 0}
        for old_idx, lbl in final_label.items():
            remap[old_idx] = label_to_new[lbl]

        lut = np.zeros(max(remap) + 1, dtype=np.uint8)
        for old_idx, new_idx in remap.items():
            lut[old_idx] = new_idx
    else:
        label_to_new = {label: idx for idx, label in new_class_map.items()}
        parent = lambda lbl: re.match(r'\d+', lbl).group()
        remap = {0: 0}
        for old_idx, label in old_class_map.items():
            remap[old_idx] = label_to_new[parent(label)]

        lut = np.zeros(max(remap) + 1, dtype=np.uint8)
        for old_idx, new_idx in remap.items():
            lut[old_idx] = new_idx
            
    return lut

def get_collapse(option,stage=None):
    """ 
    option 0 : 
        collapse to 18 classes to 3
    option 1 : 
        the training with important vessels + small branches colapsed . 12
    option 2 : has two stages . 
        stage 1 : train only big vessels 7 classes 
        stage 2 : train only for small collaped classes 6 classes (10a set to 0)
    option 3 : has two stage
        stage 1 : train for families : 7 classes
        stage 2 : train for only big classes : 7 classes
    """
    if (option == 0):
        collapse = {
            # 1 = LAD territory
            '6':   '1',
            '7':   '1',
            '8':   '1',
            '9':   '1',
            '9a':  '1',
            '10':  '1',
            '10a': '1',

            # 2 = LCx territory
            '11':  '2',
            '12':  '2',
            '12a': '2',
            '12b': '2',
            '13':  '2',
            '14':  '2',
            '14a': '2',
            '14b': '2',
            '15':  '2',

            # 3 = left main
            '5':   '3',
        }
        class_map = {
            1:"LAD",2:"LCX",3:"5"
        }
    elif(option == 1):
        class_map = {
            1:'5', 2:'12', 3:'6', 4:'9', 5:'7', 6:'9a+10', 7:'8+10a',
            8:'11', 9:'12a+12b', 10:'13', 11:'14+14a+14b+15'
        }
        collapse = {
            '5':   '1',                  
            '12':  '2',                
            '6':   '3',                    
            '9':   '4',                          
            '7':   '5',                          
            '9a':  '6', '10':  '6',                    
            '8':   '7', '10a': '7',                      
            '11':  '8',                           
            '12a': '9', '12b': '9',                   
            '13':  '10',                            
            '14':  '11', '14a': '11', '14b': '11', '15': '11',
        }
    elif(option == 2):
        if(stage == 1):
            class_map = {
                1: '5', 2: '6', 3: '7', 4: '8', 5: '11', 6: '13'
            }
            collapse = {
                '12':'0',                     
                '9':'0',                             
                '9a':'0','10':'0',                   
                '10a':'0',                        
                '12a':'0','12b':'0',            
                '14':'0','14a':'0','14b':'0','15':'0',     
                '5':'1','6':'2','7':'3','8':'4','11':'5','13':'6',
            }  
        elif(stage == 2):
            class_map = {1:'12', 2:'9', 3:'9a+10', 4:'12a+12b', 5:'14+14a+14b+15'}
            collapse = {
                '12':'1',             
                '9':'2',                  
                '9a':'3','10':'3',                  
                '10a':'0',                      
                '12a':'4','12b':'4',                
                '14':'5','14a':'5','14b':'5','15':'5',
                '5':'0','6':'0','7':'0','8':'0','11':'0','13':'0',
            }   
    elif(option == 3):
        class_map = {
            1: '5', 2: '6', 3: '7', 4: '8', 5: '11', 6: '13'
        }
        if(stage == 1):
            collapse = {
                '14': '13', 
                '14a': '13',
                '14b': '13', 
                '15': '13',
                '12a': '11', 
                '12b': '11',
                '12': '11',
                '9a': '7', 
                '10': '7',
                '10a': '8',
                "9" : "6",
                "5":"5",
                "6":"6",
                "7":"7",
                "8":"8",
                "11":"11",
                "13":"13",
            }
        elif(stage == 2):
            collapse = {
                '12':'0',                     
                '9':'0',                             
                '9a':'0','10':'0',                   
                '10a':'0',                        
                '12a':'0','12b':'0',            
                '14':'0','14a':'0','14b':'0','15':'0',     
                '5':'1','6':'2','7':'3','8':'4','11':'5','13':'6',
            }  
    return collapse , class_map