from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import albumentations as A
import copy
import cv2
import datetime
import json
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr

# %% [markdown]
# dataset
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
# %%
# %% [markdown]
# helpers
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

        
# %%
# %% [markdown]
# logger
colors = np.array([
    (242,  24,  24),   # Red
    (242,  77,  24),   # Red-Orange
    (242, 129,  24),   # Orange
    (242, 181,  24),   # Yellow-Orange
    ( 24, 242, 216),   # Cyan
    (242, 234,  24),   # Yellow
    (146,  24, 242),   # Purple
    (199, 242,  24),   # Yellow-Green
    (146, 242,  24),   # Lime
    ( 94, 242,  24),   # Green
    (242,  24, 181),   # Fuchsia
    ( 42, 242,  24),   # Green (brighter)
    ( 94,  24, 242),   # Violet
    ( 24, 242,  59),   # Spring Green
    (242,  24, 129),   # Pink
    ( 24, 242, 111),   # Aquamarine
    ( 24, 242, 164),   # Turquoise
    ( 24, 164, 242),   # Azure
    (199,  24, 242),   # Magenta
    ( 24, 216, 242),   # Sky Blue
    ( 24, 111, 242),   # Blue
    (242,  24, 234),   # Hot Pink
    ( 24,  59, 242),   # Royal Blue
    ( 42,  24, 242),   # Indigo
    (242,  24,  77),   # Rose
], dtype=np.uint8)

def save_full_report(recorder,output_base_path,model,valid_loader,
                     args,class_map,name=None):
    now = datetime.datetime.now()
    save_folder_name = str(now)
    if(name):
        save_folder_name += f" [{name}]"
    output_folder_path = os.path.join(output_base_path,save_folder_name)

    os.makedirs(output_folder_path,exist_ok=True)

    print("Saving Memory")
    save_memory(recorder,args,output_folder_path)

    print("Saving All Plots")
    draw_loss_plots(recorder , output_folder_path)
    draw_avg_metric_plots(recorder , output_folder_path)
    draw_all_metric_plots(recorder , output_folder_path)

    print("Saving Examples")
    draw_examples(model,valid_loader,args,class_map,output_folder_path)

    print("Saving Verbal Results")
    write_verbal_results(recorder,output_folder_path)

    print("Copying Notebook To Results")
    notebook_name = "nnUnetAttention.ipynb"

    notebook_out_path = os.path.join(output_folder_path,"notebook.ipynb") 
    shutil.copyfile(f"./{notebook_name}",notebook_out_path )

def write_verbal_results(recorder,output_base_path):
    report = ""
    report_path = os.path.join(output_base_path,"report.txt")
    losses_keys = recorder.losses_keys
    with open("./data/train_count.json","r") as f:
        train_count = json.load(f)
    for part,data in recorder.metric_avg_list.items():
        report +=f"======= > {part} verbal Report < =======\n"

        dice_list = data["dice"]
        precison_list = data["precision"]
        recall_list = data["recall"]

        best_idx = int(np.argmax(dice_list))
        
        best_dice = dice_list[best_idx]
        best_precision = precison_list[best_idx]
        best_recall = recall_list[best_idx]


        report += (
            f"best epoch : [{best_idx+1}]\n"
            f"best dice : [{best_dice}] - best precision : [{best_precision}] - best recall : [{best_recall}] \n"
        )
    
        for loss_name in losses_keys:
            loss_list = recorder.history[part][loss_name]
            best_loss = loss_list[best_idx]

            report += f"bset {loss_name} : [{best_loss}] - "
            

        for index , c in recorder.class_maps.items():
            dice = recorder.metric_history[part]["dice"][index][best_idx]
            precision = recorder.metric_history[part]["precision"][index][best_idx]
            recall = recorder.metric_history[part]["recall"][index][best_idx]

            counts = train_count[c]
            report += f"{c} => dice : {dice} - p : {precision} - r : {recall} || train counts : {counts}\n"
        report +="<=><=><=><=><=><=><=><=><=><=><=><=><=><=><=><=><=>\n"
    with open(report_path , "w") as f : 
        f.write(report)

def save_memory(recorder,args,output_folder_path):
    history_path = os.path.join(output_folder_path,"loss_history.json")
    full_metric_path = os.path.join(output_folder_path,"full_metric_hostory.json")
    avg_metric_path = os.path.join(output_folder_path,"avg_metric_hostory.json")
    args_path = os.path.join(output_folder_path,"args.json")

    with open(history_path , "w") as f:
        json.dump(recorder.history,f,indent=4)
    with open(full_metric_path , "w") as f:
        json.dump(recorder.metric_history,f,indent=4)
    with open(avg_metric_path , "w") as f:
        json.dump(recorder.metric_avg_list,f,indent=4)
    with open(args_path , "w") as f:
        json.dump(args,f,indent=4)

def draw_loss_plots(recorder,output_folder_path):
    plt.figure(figsize=(15,20))
    
    losses_keys = recorder.losses_keys
    colors = ["g","r","b","y","orange"]
    colors_per_class = {}

    for i,loss_name in enumerate(losses_keys):
        colors_per_class[loss_name] = colors[i]

    plt_path =os.path.join(output_folder_path,"loss_plot.png")
    
    for i,part in enumerate(recorder.history):
        plt.subplot(2,1,i+1)
        for loss_name,data in recorder.history[part].items():
            length = len(data)-1
            x = np.arange(length)
            plt.plot(x,data[:-1],color = colors_per_class[loss_name],label=loss_name)
        plt.title(f"{part} loss plot")
        plt.legend()
    plt.savefig(plt_path,dpi=150)

def draw_avg_metric_plots(recorder,output_folder_path):
    plt.figure(figsize=(15,20))
    plt_path =os.path.join(output_folder_path,"avg_metrics.png")
    for i,part in enumerate(recorder.metric_avg_list):
        plt.subplot(2,1,i+1)

        dice_data = recorder.metric_avg_list[part]["dice"]
        precision_data = recorder.metric_avg_list[part]["precision"]
        recall_data = recorder.metric_avg_list[part]["recall"]

        length = len(dice_data)
        x = np.arange(length)
        plt.plot(x,dice_data,color="g",label="dice")
        plt.plot(x,precision_data,color="r",label="precision")
        plt.plot(x,recall_data,color="b",label="recall")
        plt.title(f"{part} avg dice plot")
        plt.legend()
    plt.savefig(plt_path)
def draw_all_metric_plots(recorder,output_folder_path):
    for part in recorder.history: 
        plt_path =os.path.join(output_folder_path,f"{part}_full_metric.png")
        plt.figure(figsize=(30,30))
        for i , class_index  in enumerate(recorder.metric_history[part]["dice"]):
            dice_data = recorder.metric_history[part]["dice"][class_index]
            precision_data = recorder.metric_history[part]["precision"][class_index]
            recall_data = recorder.metric_history[part]["recall"][class_index]

            class_name = recorder.class_maps[class_index]
            plt.subplot(5,5,i+1)
            length = len(dice_data)
            x = np.arange(length)
            plt.plot(x,dice_data,color="g",label="dice")
            plt.plot(x,precision_data,color="r",label="precision")
            plt.plot(x,recall_data,color="b",label="recall")
            plt.title(f"{class_name}")
            plt.legend()

        plt.savefig(plt_path)



@torch.no_grad()
def draw_examples(model,valid_loader,args,class_map,output_folder_path,w=6,h=6):
    plt_path = os.path.join(output_folder_path,"examples.png")
    plt.figure(figsize=(30,30))
    plot_count =18
    patches = [
        mpatches.Patch(color=np.array(colors[j-1]) / 255.0, label=class_map[j])
        for j in range(1,len(class_map)+1)
    ]
    i=0
    img_index=1
    valid_iterator = iter(valid_loader)
    for i in tqdm(range(plot_count)):
        img , mask = next(valid_iterator)
        with torch.autocast(device_type=args["device"],dtype=torch.float16):
            pred_mask = model(img.to(args["device"]))
        pred_mask = pred_mask.cpu().numpy()
        pred_mask = np.argmax(pred_mask,axis=1)
        mask = mask.numpy()
        img = img.numpy()
        x_disp = (img[0,0] - img[0,0].min()) / (img[0,0].max() - img[0,0].min() + 1e-8)
        rgb = np.repeat(x_disp[..., None], 3, axis=2)*255
        real_annoted = draw_mask(rgb,mask[0],args,colors)
        pred_annoted = draw_mask(rgb,pred_mask[0],args,colors)
        
        plt.subplot(h,w,img_index)
        plt.imshow(real_annoted)
        plt.title(f"Ground Truth ")
        plt.subplot(h,w,img_index+1)
        plt.imshow(pred_annoted)
        plt.title(f"Predicted ")
        img_index+=2
        if(img_index-1==h):
            plt.legend(
                handles=patches,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.,
                title="Classes"
            )
        i+=1
    plt.savefig(plt_path)
# %%
# %% [markdown]
# preprocessing
class CLAHE : 
    def __init__(self,clipLimit=2.0,tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    def __call__(self,img):
        enhanced = np.clip(img, 0, 255)
        return self.clahe.apply(enhanced)
class WhiteTopHat:
    def __init__(self,kernel_size = (50, 50),turn_neg = True):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.turn_neg = turn_neg
    def __call__(self,img):
        neg_img = cv2.bitwise_not(img)
        tophat_img = cv2.morphologyEx(neg_img, cv2.MORPH_TOPHAT, self.kernel,borderType=cv2.BORDER_REPLICATE)
        # tophat_img = morphology.white_tophat(neg_img, self.kernel) 
        return cv2.subtract(img, tophat_img)
        
# Augementations 
def normalize_xca(img, **kwargs):
    x = img.astype(np.float32, copy=False)
    m = x > 0
    if np.any(m):
        mean = x[m].mean()
        std  = x[m].std()
        x[m] = (x[m] - mean) / (std + 1e-8)
        x[~m] = 0.0
    else:
        x = x / 1.0
    return x
# %%
# %% [markdown]
# recorder
class HistoryRecorder:
    def __init__(self,class_maps,losses_keys,class_count = 25):

        self.history = {
            "train":{},
            "valid":{}
        }
        for key in losses_keys:
            self.history["train"][key] = [[]]
            self.history["valid"][key] = [[]]
        self.losses_keys = losses_keys
        self.metric_history={}
        self.class_maps =class_maps
        for part in self.history :
            self.metric_history[part]={"dice":{},"precision":{},"recall":{}}
            for i in range(1,class_count+1):
                self.metric_history[part]["dice"][i]=[]
                self.metric_history[part]["recall"][i]=[]
                self.metric_history[part]["precision"][i]=[]
                
        self.metric_avg_list = {
            "train":{"dice":[],"precision":[],"recall":[]},
            "valid":{"dice":[],"precision":[],"recall":[]}
        }

        self.class_count = class_count 
        
    def add_losses(self,part,loss_dict):
        for loss_name,loss in loss_dict.items():
            self.history[part][loss_name][-1] += [loss]
        
    def add_metrics(self,dice,precision,recall,part):
        dice = dice[1:]
        precision = precision[1:]
        recall = recall[1:]
        for i in range(self.class_count):
            d = dice[i]
            r = recall[i]
            p = precision[i]
            self.metric_history[part]["dice"][i+1].append(d)
            self.metric_history[part]["recall"][i+1].append(r)
            self.metric_history[part]["precision"][i+1].append(p)
    def avg_losses(self,part):
        for key in self.history[part]:
            self.history[part][key][-1] = np.mean(self.history[part][key][-1])
            self.history[part][key].append([])
            
    def print_loss_report(self,part,epoch,avg_first=True):
        if(avg_first):
            self.avg_losses(part)
            
        report = f"{part} ==> epcoh ({epoch})\n"
        co=0
        for loss_name in self.losses_keys:
            loss_list = self.history[part][loss_name]

            loss = loss_list[-2]
            report += f"{loss_name} : {loss}"
            if((co+1)%3==0):
                report += "\n"
            else:
                report+=" - "
            co+=1  
                 
        print(report)
    def print_metrics_report(self,part,epoch,class_wise=False):
        
        report_temp = f"{part} avg metrics for epoch {epoch} :\n"
        report_class_wise_temp = ""
        avg_dice = 0
        avg_precision = 0
        avg_recall = 0
        for index , c in self.class_maps.items():
            dice = self.metric_history[part]["dice"][index][-1]
            precision = self.metric_history[part]["precision"][index][-1]
            recall = self.metric_history[part]["recall"][index][-1]
            avg_dice += dice
            avg_precision += precision
            avg_recall += recall
            if(class_wise):
                report_class_wise_temp += f"{c} => dice : {dice} p : {precision} , r : {recall}\n"
        
        avg_dice = avg_dice/self.class_count
        avg_precision = avg_precision/self.class_count
        avg_recall = avg_recall/self.class_count

        self.metric_avg_list[part]["dice"]+=[avg_dice]
        self.metric_avg_list[part]["precision"]+=[avg_precision]
        self.metric_avg_list[part]["recall"]+=[avg_recall]
        
        report_temp+=f"avg dice : {avg_dice} - avg precision : {avg_precision} - avg recall : {avg_recall}"

        if(class_wise):
            report_temp = report_temp + "\n" +report_class_wise_temp[:-1] #removing last \n
        print(report_temp)




# %%
# %% [markdown]
# nnunet_blocks
class Conv(nn.Module):
    def __init__(self,in_c , out_c,p):
        super(Conv,self).__init__()
        self.layers =  nn.Sequential( 
            nn.Conv2d(
                in_channels = in_c , 
                out_channels = out_c ,
                kernel_size=3, 
                stride = 1 ,
                padding = p
            ),
            nn.InstanceNorm2d(out_c, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
    def forward(self,x):
        return self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self,in_c , out_c,p=1):
        super(EncoderBlock,self).__init__()
        self.layers = nn.Sequential(
            Conv(in_c = in_c , out_c = out_c , p=p),
            Conv(in_c = out_c , out_c = out_c , p=p)   
        )
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        z = self.layers(x)
        return z , self.pool(z)

class DecoderBlock(nn.Module):
    def __init__(self,in_c ,out_c , f_int_scale,gate_c = None , attention=False):
        super(DecoderBlock,self).__init__()
        self.conv1 = Conv(in_c=in_c , out_c=out_c,p=1)
        self.conv2 = Conv(in_c = out_c , out_c = out_c , p=1)
        self.upsampler = nn.ConvTranspose2d(
            in_channels = out_c , 
            out_channels = out_c//2 ,
            kernel_size=2 ,
            stride=2
        )
        #in_c * 2 = gate_c
        if(attention):
            self.gate = AttentionGate(gate_in_c=gate_c ,f_int_scale=f_int_scale,skip_in_c=in_c//2)
        self.attention=attention
    def forward(self,x_in,x_skip,x_gate):
        if(self.attention):
            x_skip = self.gate(x_skip , x_gate)
        if(x_in.shape[2] != x_skip.shape[2] or x_in.shape[3] != x_skip.shape[3]):
            x_skip = crop_dims(x_in,x_skip)
        
        x = torch.cat([x_skip,x_in],dim=1)
        z = self.conv1(x)
        gate_z = self.conv2(z)
        upsampled_z = self.upsampler(gate_z)
        if(self.attention):
            return upsampled_z , gate_z
        return upsampled_z , None

class BottleNeck(nn.Module):
    def __init__(self,in_c , out_c,p,attention=False):
        super(BottleNeck,self).__init__()
        self.conv1 = Conv(in_c=in_c , out_c=out_c,p=p)
        self.conv2 = Conv(in_c = out_c , out_c = out_c , p=p)
        self.upsampler = nn.ConvTranspose2d(
            in_channels = out_c , 
            out_channels = out_c//2 ,
            kernel_size=2 ,
            stride=2
        )
        self.attention=attention
    def forward(self,x):
        z = self.conv1(x)
        gate_z = self.conv2(z)
        upsampled_z = self.upsampler(gate_z)
        if(self.attention):
            return upsampled_z , gate_z
        return upsampled_z , None

class AttentionGate(nn.Module):
    def __init__(self,gate_in_c,skip_in_c,f_int_scale=2,f_int=None,scaler="sigmoid"):
        super(AttentionGate,self).__init__()
        f_int = min(gate_in_c//f_int_scale,skip_in_c//f_int_scale) if f_int==None else f_int
        f_int = 1 if f_int == 0 else f_int
        self.conv_gate = nn.Conv2d(in_channels = gate_in_c , out_channels = f_int , 
                                   kernel_size = 1)
        self.conv_skip = nn.Conv2d(in_channels = skip_in_c , out_channels = f_int , 
                                   kernel_size = 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_shrink = nn.Conv2d(in_channels = f_int , out_channels = 1 ,
                                     kernel_size = 1)
        if(scaler =="sigmoid"):
            self.scaler = nn.Sigmoid()    
    def forward(self,x_skip,x_gate):
        x_gate_int = self.conv_gate(x_gate)
        x_skip_int = self.conv_skip(x_skip)
        # x_skip_int = crop_dims(x_gate_int,x_skip_int)
        if x_skip_int.shape[2:] != x_gate_int.shape[2:]:
            x_skip_int = F.interpolate(
                x_skip_int, 
                size=x_gate_int.shape[2:], 
                mode="bilinear", 
                align_corners=False
            )
        
        added_x = x_skip_int + x_gate_int
        relu_x = self.relu(added_x)
        shrinked_x = self.conv_shrink(relu_x)
        sig_x = self.scaler(shrinked_x)
        # padded_x = padd_dims(x_skip , sig_x)
        if sig_x.shape[2:] != x_skip.shape[2:]:
            padded_x = F.interpolate(
                sig_x, 
                size=x_skip.shape[2:], 
                mode="bilinear", 
                align_corners=False
            )

        return padded_x*x_skip

class Head(nn.Module):
    def __init__(self,in_c ,out_c ,class_count ,f_int_scale, gate_c = None , attention=False):
        super(Head,self).__init__()
        self.conv1 = Conv(in_c=in_c , out_c=out_c,p=1)
        self.conv2 = Conv(in_c = out_c , out_c = out_c , p=1)
        self.conv1x1 = nn.Conv2d(
            in_channels = out_c , 
            out_channels = class_count ,
            kernel_size=1
        )
        
        if(attention):
            self.gate = AttentionGate(gate_in_c=gate_c , f_int_scale=f_int_scale,skip_in_c=in_c//2)
        self.attention=attention
    def forward(self,x_in,x_skip,x_gate):
        if(self.attention):
            x_skip = self.gate(x_skip , x_gate)
        if(x_in.shape[2] != x_skip.shape[2] or x_in.shape[3] != x_skip.shape[3]):
            x_skip = crop_dims(x_in,x_skip)
            
        x = torch.cat([x_skip,x_in],dim=1)
        z = self.conv1(x)
        gate_z = self.conv2(z)
        class_feature_maps = self.conv1x1(gate_z) 
        
        return class_feature_maps
# %%
# %% [markdown]
# nnunet
class nnUnet(nn.Module):
    def __init__(self,args,encoder_channel_settings=None,decoder_channel_settings=None):
        super(nnUnet,self).__init__()
        
        in_c = args["in_c"]
        class_count = args["class_count"]
        attention = args["attention"]
        image_shape = args["image_shape"]
        base_channel = args["base_channel"]
        f_int_scale = args["f_int_scale"]
        max_channels = args["max_channels"]
        input_channels = args["input_channels"]
        h = image_shape[0]
        w = image_shape[1]
        
        max_pool_count = 0
        co=0
        
        while(w>4 and h>4):
            w/=2
            h/=2
            co+=1
        print(f"number of layers : {co}")

        # create encoder settings 
        if(encoder_channel_settings is None):
            self.encoder_channel_settings = [base_channel]
            for i in range(co-1):
                new_c =min(self.encoder_channel_settings[i]*2,max_channels)
                self.encoder_channel_settings +=[new_c]
        else :
            self.encoder_channel_settings = encoder_channel_settings
        
        # create bottleneck settings
        self.bottle_neck_channel_setting = self.encoder_channel_settings[-1]*2
        # create decoder settings 
        if(decoder_channel_settings is  None):
            self.decoder_channel_settings =[]
            for i in range(co-1):
                self.decoder_channel_settings = [self.encoder_channel_settings[i]*2] +  self.decoder_channel_settings
        else :
            self.decoder_channel_settings = decoder_channel_settings

        
        # build encoder
        self.encoders = nn.ModuleList()
        for i in range(co):
            output_channels = self.encoder_channel_settings[i]
            self.encoders.append(EncoderBlock(in_c=input_channels,out_c=output_channels , p=1))
            input_channels = output_channels
        # build bottleneck
        self.bottle_neck = BottleNeck(in_c = output_channels ,out_c = self.bottle_neck_channel_setting , p=1,attention = attention)
        #build decoder
        input_channels = self.bottle_neck_channel_setting
        self.decoders = []
        for i in range(co-1):
            output_channels = self.decoder_channel_settings[i]
            self.decoders = [
                DecoderBlock(
                    in_c = input_channels , 
                    out_c=output_channels , 
                    gate_c = input_channels , 
                    attention = attention,
                    f_int_scale=f_int_scale
                )] + self.decoders
            input_channels = output_channels

        self.decoders = nn.ModuleList(self.decoders)
        self.attention = attention
        
        self.head = Head(
            in_c = input_channels , 
            out_c=input_channels//2 ,
            class_count = class_count,
            gate_c = input_channels , 
            attention = False,
            f_int_scale=f_int_scale
        )
        print("encoder settings : ", self.encoder_channel_settings)
        print("bottle-neck settings : ", self.bottle_neck_channel_setting)
        print("decoder settings : ", self.decoder_channel_settings)
        print("head settings : ",class_count)
    def forward(self,x):
        skips = []
        for encoder in self.encoders : 
            skip , out = encoder(x)
            skips += [skip]
            x = out
        x_in,gate_in = self.bottle_neck(x)
        
        for i in range(len(self.decoders) - 1, -1, -1):
            decoder = self.decoders[i]
            skip = skips[i+1]
            x_out,gate_out = decoder(x_in,skip,gate_in)
            x_in=x_out
            gate_in=gate_out
        # print(x_in.shape)
        return self.head(
            x_in = x_in,
            x_skip = skips[0],
            x_gate = gate_in
        )
# args = {
#     "base_path" : "../arcade/nnUnet_dataset/syntax",
#     "in_c" : 1,
#     "base_channel" :32,
#     "image_shape" : (512,512),
#     "class_count" : 26 ,
#     "attention" : True,
#     "batch_size" : 5,
#     "num_workers" : 4,
#     "device" : "cuda" if torch.cuda.is_available() else "cpu",
#     "lr" : 0.0001,
#     "momentum" : 0.9,
#     "epcohs":1,
#     "f_int_scale" : 2,
#     "full_report_cycle" : 10,
#     "max_channels":512,
#     "input_channels":1,
#     "output_base_path" : "./outputs",
#     "name" : "NoAttention"
# }
# class_map = {
#     0: '1',1: '2', 2: '3',3: '4',
#     4: '5',5: '6',6: '7',7: '8',
#     8: '9',9: '9a',10: '10',11: '10a',
#     12: '11',13: '12',14: '12a',15: '13',
#     16: '14',17: '14a',18: '15',19: '16',
#     20: '16a',21: '16b',22: '16c',
#     23: '12b',24: '14b'
# }
# model = nnUnet(args)
# ls = torch.ones((1,1,512,512)).float()
# print(model(ls).shape)
# %%
# %% [markdown]
# losses
class UnetLoss(nn.Module):
    def __init__(self,args,eps = 1e-8):
        super(UnetLoss,self).__init__()
        self.class_count = args["class_count"]
        self.loss_type = args["loss_type"]
        self.bce_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.eps = eps
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.gamma = args["gamma"]
        self.sum_dims = (2,3)
        if(self.loss_type=="dice loss"):
            print("loss is set to dice")
            self.loss_fn = DiceLoss(self.eps,self.sum_dims)
        elif(self.loss_type=="tversky loss"):
            print("loss is set to tversky")
            self.loss_fn = TverskyLoss(self.eps,self.sum_dims,self.alpha,self.beta,self.gamma)

    def forward(self,pred_mask , gt_mask):
        
        # Cross Entropy Loss
        ce_loss = self.bce_fn(pred_mask, gt_mask)
        # Dice Loss
        onehot_mask = F.one_hot(gt_mask, num_classes=self.class_count)
        onehot_mask = onehot_mask.permute(0, 3, 1, 2).float()  

        prob = self.softmax(pred_mask)

        forground_prob = prob[:,1:]
        forground_onehot_mask = onehot_mask[:,1:]
        present_class = forground_onehot_mask.sum(dim=self.sum_dims)>0
        
        second_loss = self.loss_fn(
            pred_probs = forground_prob,
            gt = forground_onehot_mask,
            present_class=present_class
        )
        total_loss = ce_loss + second_loss

        loss_dict = {
            "CE loss" : ce_loss,
            self.loss_type : second_loss
        }
        return total_loss , loss_dict

class DiceLoss(nn.Module):
    def __init__(self,eps,sum_dims):
        super(DiceLoss,self).__init__()
        self.eps = eps
        self.sum_dims = sum_dims
    def forward(self,pred_probs,gt,present_class=None):
        dice_numerator = (gt * pred_probs).sum(dim=self.sum_dims)
        dice_denominator = gt.sum(dim=self.sum_dims) + pred_probs.sum(dim=self.sum_dims)
        per_class_dice_loss = (2*dice_numerator)/(dice_denominator + self.eps)
        if(present_class is None):
            dice_loss = -per_class_dice_loss.mean()
        else:
            dice_loss = -per_class_dice_loss[present_class].mean()
        return dice_loss

class TverskyLoss(nn.Module):
    def __init__(self,eps,sum_dims,alpha=0.3,beta=0.7,gamma=1.33):
        super(TverskyLoss,self).__init__()
        self.eps = eps
        self.sum_dims = sum_dims
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    def forward(self,pred_probs,gt,present_class=None):
        tp = (gt * pred_probs).sum(dim=self.sum_dims)
        fp = ((1-gt)*pred_probs).sum(dim=self.sum_dims)
        fn = ((1-pred_probs)*gt).sum(dim=self.sum_dims)
        t_index = (tp + self.eps) / (tp + self.alpha*fp + self.beta*fn + self.eps) 
        if(present_class is None):
            t_index = t_index.mean()
        else:
            t_index = t_index[present_class].mean()
        return (1 - t_index)**self.gamma 

    
    
# %%
# %% [markdown]
# trainer
def train_fn(model,img,gt_mask,optimizer,loss_fn,scaler,args):
    optimizer.zero_grad()
    with torch.autocast(device_type=args["device"],dtype=torch.float16):
        pred_mask = model(img)
        loss , loss_dict = loss_fn(pred_mask , gt_mask)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    loss = loss.detach().cpu().item()
    for loss_name in loss_dict:
        loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
    
    loss_dict["total loss"] = loss
    return loss_dict , pred_mask
    
def trainer(args,recorder,model,optimizer,loss_fn,train_loader,valid_loader):
    device = args["device"]
    epcohs = args["epcohs"]
    class_count = args["class_count"]
    full_report_cycle = args["full_report_cycle"]
    scaler = torch.amp.GradScaler(device = device) 
    for ep in tqdm(range(epcohs)):
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)
        
        model.train()
        class_wise_report = False
        for img , gt_mask in tqdm(train_loader) : 
            gt_mask = gt_mask.long()
            img = img.to(device)
            gt_mask = gt_mask.to(device)
            loss_dict , pred_mask = train_fn(
                model = model,
                img = img,
                gt_mask = gt_mask,
                optimizer = optimizer,
                loss_fn = loss_fn,
                scaler = scaler,
                args = args
            )
            
            TP , _ , FP , FN = TP_TN_FP_FN(pred_mask.detach(),gt_mask.detach(),process_preds=True)
            total_TP += TP
            total_FP += FP
            total_FN += FN
            
            recorder.add_losses("train",loss_dict)
            
        dice_score = (2 * total_TP + 1e-8) / (2 * total_TP + total_FP + total_FN + 1e-8)
        precision = total_TP /(total_FP + total_TP + 1e-8) 
        recall = total_TP /(total_FN + total_TP + 1e-8) 
        
        recorder.add_metrics(
            dice_score.tolist(),
            precision.tolist(),
            recall.tolist(),
            part = "train"
        )
        
        recorder.print_loss_report("train",ep)
        recorder.print_metrics_report("train",ep,class_wise=False)
        print("<=>"*20)
        
        if((ep+1)%full_report_cycle==0):
            class_wise_report=True
            
        evaluation(
            recorder=recorder,
            model=model,
            loss_fn=loss_fn,
            valid_loader=valid_loader,
            class_wise_report=class_wise_report,
            class_count = class_count,
            epoch=ep,
            device=device)
            
@torch.no_grad()
def evaluation(recorder,model,loss_fn,valid_loader,class_count,class_wise_report=False,epoch=None,device="cuda"):
    model.eval()
    total_TP = total_TP = torch.zeros(class_count)
    total_FP = torch.zeros(class_count)
    total_FN = torch.zeros(class_count)
    
    for img , gt_mask in valid_loader:
        gt_mask = gt_mask.long()
        img = img.to(device)
        gt_mask = gt_mask.to(device)
        with torch.autocast(device_type=device,dtype=torch.float16):
            pred_mask = model(img)
            loss , loss_dict = loss_fn(pred_mask , gt_mask)

            loss = loss.detach().cpu().item()
            for loss_name in loss_dict:
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().item()
        
            loss_dict["total loss"] = loss
        
        TP , _ , FP , FN = TP_TN_FP_FN(pred_mask,gt_mask,process_preds=True)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        
        recorder.add_losses("valid",loss_dict)
        
    dice_score = (2 * total_TP + 1e-8) / (2 * total_TP + total_FP + total_FN + 1e-8)
    precision = total_TP /(total_FP + total_TP + 1e-8) 
    recall = total_TP /(total_FN + total_TP + 1e-8) 

    recorder.add_metrics(
        dice_score.tolist(),
        precision.tolist(),
        recall.tolist(),
        part = "valid"
    )
    recorder.print_loss_report("valid",epoch)
    recorder.print_metrics_report("valid",epoch,class_wise=class_wise_report)
    print("-"*60)
# %%