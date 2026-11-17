import torch
import pandas as pd
import albumentations as A
from pathlib import Path
import os
from tqdm.auto import tqdm
from .helpers import TP_TN_FP_FN
import torch.nn.functional as F
import cv2
from .preprocessing import (
    normalize_xca,normalize_empty,get_preprocessing,
    remove_black_border_multithread,BrightnessMultiplicativeNNUNet2D,
    ContrastAugmentationNNUNet2D
)

transforms = {
    "Affine": A.Affine(
        translate_percent=(-0.05, 0.05), # Replaces shift_limit
        scale=(0.9, 1.1),                # Replaces scale_limit
        rotate=(-10, 10),                # Replaces rotate_limit
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=0,
        p=0.5
    ),
    "Brightness": A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.25,
        p=1.0,
    ),
    "Gamma": A.RandomGamma(
        gamma_limit=(70, 140),
        p=1.0,
    ),
    "GaussNoise": A.GaussNoise(
        std_range=(0.01, 0.05),
        mean_range=(0.0, 0.0),
        per_channel=False,
        p=1.0,
    ),
    "GaussianBlur": A.GaussianBlur(
        blur_limit=(3, 5),
        sigma_limit=(0.1, 1.0),
        p=1.0,
    ),
    "MotionBlur": A.MotionBlur(
        blur_limit=(3, 5),
        p=1.0,
    ),
    "ElasticTransform":  A.ElasticTransform(
                    alpha=300.0,
                    sigma=8.0,         
                    p=1.0
                ),
    "GridDistortion": A.GridDistortion(
                    num_steps=5, 
                    distort_limit=(-0.5,0.5),  
                    p=1.0
                ),
    "OpticalDistortion":A.OpticalDistortion(
                    distort_limit=(-0.4, 0.4),  
                    # shift_limit removed entirely
                    p=1.0
                ),
    # "HorizontalFlip" : A.HorizontalFlip(p=1),
    # "VerticalFlip" : A.VerticalFlip(p=1),
}

class ParallelCompose:
    """A custom wrapper to run multiple Albumentations transforms in parallel."""
    def __init__(self, transforms_dict,normalizer):
        # Wrap each standalone transform in its own A.Compose
        self.pipelines = {
            name: A.Compose(
                [transform],
                additional_targets={'v_mask': 'mask',"expert_mask":"mask"}
            ) 
            for name, transform in transforms_dict.items()
        }
        self.pipelines["raw"] =  A.Compose(
            [],
            additional_targets={'v_mask': 'mask',"expert_mask":"mask"}
        ) 
        self.normalizer = normalizer
    def __call__(self, **kwargs):
        # Run the input through every pipeline and store the results in a dict
        res = {}
        for name,t in self.pipelines.items():
            results =  t(**kwargs)
            img = results["image"]
            img = self.normalizer(img)["image"]
            mask = results["mask"][...,0]

            
            img = torch.from_numpy(img).permute(2,0,1)
            mask = torch.from_numpy(mask).long()

            img = img.unsqueeze(0)
            mask = mask.unsqueeze(0)
            res[name] = {"image":img,"mask":mask}

        
        return res
    
def aug_analysis(
        best_model,
        images,
        normalizer,
        args,
        model_name,
        dice_loss_fn=None,
        out_folder="model_analysis"):
    
    class_count = args["class_count"]
    base_ds_path = Path(args["paths"]["base_path"])
    save_root = base_ds_path / out_folder / model_name / "aug_analysis"
    os.makedirs(save_root, exist_ok=True)

    parallel_pipeline = ParallelCompose(transforms,normalizer=normalizer)
    with torch.no_grad():
        for part , image_list in images.items():
            part_results = {"names":[],"raw_d":[],"raw_r":[],"raw_p":[]}
            if(dice_loss_fn is not None):
                part_results["raw_dice_loss"] = []
            for key in transforms:
                part_results[f"{key}_d"]=[]
                part_results[f"{key}_r"]=[]
                part_results[f"{key}_p"]=[]
                if(dice_loss_fn is not None):
                    part_results[f"{key}_dice_loss"] = []
            for idx,(img,hard_idx,label,skel_mask,side,name_stem) in enumerate(tqdm(image_list)):
                batch = parallel_pipeline(image=img[...,None],mask=label[...,None])
                for t_name , b in batch.items():
                    img = b["image"].cuda()
                    mask = b["mask"].cuda()
                    
                    with torch.autocast(device_type="cuda",dtype=torch.float16):
                        pred_mask = best_model(img)
                        pred_mask = pred_mask[0]
                        
                    TP , _ , FP , FN = TP_TN_FP_FN(pred_mask,mask,process_preds=True)
                    dice_score = (2 * TP + 1e-8) / (2 * TP + FP + FN + 1e-8)
                    precision = TP /(FP + TP + 1e-8) 
                    recall = TP /(FN + TP + 1e-8) 

                    dice_score = dice_score[1].item()
                    precision = precision[1].item()
                    recall = recall[1].item()
                    
                    part_results[f"{t_name}_d"].append(dice_score)
                    part_results[f"{t_name}_p"].append(precision)
                    part_results[f"{t_name}_r"].append(recall)
                    if(dice_loss_fn is not None):
                        onehot_gt_masks = F.one_hot(mask, num_classes=class_count)
                        onehot_gt_masks = onehot_gt_masks.permute(0, 3, 1, 2).float()
                        pred_probs = F.softmax(pred_mask,dim=1)

                        fg_pred_probs = pred_probs[:,1:]
                        fg_onehot_gt_masks = onehot_gt_masks[:,1:]
                      
                        dice_family_loss = dice_loss_fn(
                            pred_probs = fg_pred_probs,
                            gt = fg_onehot_gt_masks
                        )
                        
                        part_results[f"{t_name}_dice_loss"].append(dice_family_loss.item())
                part_results["names"].append(name_stem)
            # for p , l in part_results.items():
            #     print(p,len(l))
            part_results = pd.DataFrame(part_results)
            part_results.to_csv(save_root/f"./{part}_aug_analysis.csv",index=False)