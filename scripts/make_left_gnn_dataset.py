import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import albumentations as A
import cv2
from pathlib import Path
import os
import json
from tqdm import tqdm
import gc
from collections import Counter
import heapq
import shutil
###IE###
from models.swin_encoder import SwinUperNet,SwinUperNet2
from models.smp_models import DeepLabV3Plus
from models.S3_DSCNet_pro import DSCNet_pro
from utils.helpers import load_model,read_images,make_result_dataloaders,left_mapping_function,make_training_dataloaders
from utils.gnn_dataset_generation_left_args import (
    args,old_class_map,class_map,lut,validator_class_maps,
    root_locations,index_maps,reversed_map,root_idx_dict,
    dataset_keys,cnn_classes,extra_index_indx,collapse
)
from utils.option2_collaps import combine
from utils.graph_extraction import (
    generate_robust_mask,
    create_gnn_dataset_parallel_limited
)
from utils.k_step_graph_extraction_left import generate_dataset
from utils.gnn_mask_drawing import draw_gnn_masks
from utils.gnn_helpers import load_set
from utils.gnn_evaluation import evaluate_merged
###SS###
class EnsembledModels(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()
        self.models = torch.nn.ModuleList([load_model(model_class= c, path=p,device="cuda",model_inputs={}) for p,c in paths])
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def forward(self, img, gt_mask=None,ep=None):
        probs = torch.stack([
            F.softmax(m(img)[0][0].float(), dim=1) for m in self.models
        ]).mean(0)
        # binary_preds = self.binary_model(img)[0][0]
        # binary_preds = torch.argmax(binary_preds,dim=1)
        # return LOG-probs: argmax is unchanged, and a downstream softmax is a
        # no-op that recovers the averaged probs. Drop-in for your eval code.
        return [torch.log(probs.clamp_min(1e-8))],0

class Temp(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = load_model(
            model_class=SwinUperNet,
            path="./outputs/2026-08-11_02-30-49 [model[SwinUperNet]-multi-left-7class-50epo]",
            model_inputs={},
            device="cuda",
        )
        for p in self.model.parameters():      
            p.requires_grad = False
        self.model.eval()       

        self.second_model = SwinUperNet(args)

        self.fuse = torch.nn.Conv2d(3 + 7, 3, kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()           
        return self

    def forward(self, x, gt_mask=None, ep=None):
        with torch.no_grad():
            big_out, _ = self.model(x)     
            big_logits = big_out[0].detach()
            big_probs = F.softmax(big_logits,dim=1)
        fused = self.fuse(torch.cat([x, big_probs], dim=1))
        return self.second_model(fused)
    
class CombinedModel(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()
        self.model =load_model(model_class= paths[0][1], path=paths[0][0],device="cuda",model_inputs={})
    @torch.no_grad()
    def forward(self, img, gt_mask=None,ep=None):
        with torch.autocast("cuda"):
            big_out, _ = self.model.model(img)           # stage 1 (frozen)
            small_out, _ = self.model(img)               # Temp -> stage 2
        out,probs = combine(big_out[0], small_out[0], mode="confidence")
        return [out],probs
    
def get_images(args,collapse,old_class_map):
    train_images,sampler_weights,last_boot_level = read_images(
        paths = args["paths"],
        part = "train",
        chosen_labels=None,
        hard_classes = [],
        bootstrap = args["bootstrap"],
        training_mode = args["training_mode"],
        binary_mode = args["binary_training_mode"],
        side_mode = args["side_mode"],
        in_c = args["in_c"],
        cut_borders = args["cut_borders"],
        class_count = args["class_count"],
        collapse=collapse,
        old_class_map=old_class_map

    )
    valid_images = read_images(
        paths = args["paths"],
        part = "val",
        chosen_labels=None,
        bootstrap = args["bootstrap"],
        training_mode = args["training_mode"],
        binary_mode = args["binary_training_mode"],
        side_mode = args["side_mode"],
        in_c = args["in_c"],
        cut_borders = args["cut_borders"],
        class_count = args["class_count"],
        collapse=collapse,
        old_class_map=old_class_map
    )
    test_images = read_images(
        paths = args["paths"],
        part = "test",
        chosen_labels=None,
        bootstrap = args["bootstrap"],
        training_mode = args["training_mode"],
        binary_mode = args["binary_training_mode"],
        side_mode = args["side_mode"],
        in_c = args["in_c"],
        cut_borders = args["cut_borders"],
        class_count = args["class_count"],
        collapse=collapse,
        old_class_map=old_class_map
    )

    return train_images , valid_images , test_images
## binary model

if __name__ == "__main__":
    
    normalizer = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    )

    transforms = {
        "affine_":A.Compose([A.Affine(
            translate_percent=(-0.10, 0.10),
            scale=(0.8, 1.25),    
            rotate=(-30, 30),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,    
            p=0.5,
        )],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"}),
        "raw_":A.Compose([],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"}),
        
    }

    # paths = [  
    # ["./outputs/2026-07-28_13-46-04 [model[SwinUperNet2]-multi-left-7class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNet2],
    # ["./outputs/2026-07-27_21-19-14 [model[SwinUperNet2]-multi-left-7class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNet2],
    # ["./outputs/2026-07-28_15-57-59 [model[SwinUperNet2]-multi-left-7class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNet2],
    # ["./outputs/2026-07-28_03-08-21 [model[SwinUperNet2]-multi-left-7class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNet2],
    # ["./outputs/2026-07-27_19-09-51 [model[SwinUperNet2]-multi-left-7class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNet2],

    # ["./outputs/2026-08-03_03-14-25 [model[SwinUperNet2]-multi-left-7class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNet2],
    # ["./outputs/2026-08-02_23-27-45 [model[SwinUperNet2]-multi-left-7class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNet2]
    # ]
    # train_paths = {
    #     # "ep[5]_model[swin]":["./outputs/2026-07-19_16-07-10 [model[SwinUperNetCascade]-multi-right-6class-5epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
    #     "ep[50]_model[swin]":["./outputs/2026-08-03_02-06-25 [model[SwinUperNet2]-multi-left-7class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNet2],
    #     # "ep[15]_model[swin]":["./outputs/2026-07-19_16-07-10 [model[SwinUperNetCascade]-multi-right-6class-5epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
    #     # "ep[50]_model[deeplab]":["./outputs/2026-07-19_14-27-22 [model[DeepLabV3Plus]-multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",DeepLabV3Plus],
    # }
    
    paths = [  
        ["./outputs/2026-08-14_00-05-23 [model[SwinUperNet]-multi-left-4class-50epo]",SwinUperNet],
    ]
    # train_paths = {
    #     # "ep[5]_model[swin]":["./outputs/2026-07-19_16-07-10 [model[SwinUperNetCascade]-multi-right-6class-5epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
    #     "ep[20]_model[swin]":["./outputs/2026-08-12_13-01-41 [model[Temp]-multi-left-6class-20epo]",Temp],
    #     # "ep[15]_model[swin]":["./outputs/2026-07-19_16-07-10 [model[SwinUperNetCascade]-multi-right-6class-5epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
    #     # "ep[50]_model[deeplab]":["./outputs/2026-07-19_14-27-22 [model[DeepLabV3Plus]-multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",DeepLabV3Plus],
    # }
    class_count = len(class_map) + 1

    output_name = paths[0][0].split(" ")[0].split("/")[-1]
    base_input_path       = Path("./dataset/syntax/labels")
    base_output_path      = Path(f"./dataset/syntax/gnn_dataset/left_dataset_{output_name}_{len(class_map)}")

    

    train_images , valid_images , test_images = get_images(args,collapse,old_class_map)

    val_model = load_model(
        model_class= paths[0][1], 
        path= paths[0][0],
        device="cuda",model_inputs={}
    )
    val_model.eval()

    # train_model = CombinedModel(paths=[train_paths["ep[20]_model[swin]"]])
    # train_model = train_model.cuda()
    # train_model.eval()
    # train_model = load_model(
    #     model_class= train_paths["ep[50]_model[swin]"][1], 
    #     path=train_paths["ep[50]_model[swin]"][0],

    #     device="cuda",model_inputs={})
    
    # train_model.eval()
    if base_output_path.exists() and base_output_path.is_dir():
        shutil.rmtree(base_output_path)
    train_folder = Path("./dataset/syntax/multi_k_fold_preds/2026-08-14_14-57-19")
    train_names = os.listdir(train_folder/"pred_probs")
    train_names = [Path(name).stem for name in train_names]
    
    co = {i:{n:0 for key,n in old_class_map.items()} for i in range(1,len(class_map)+1)}
    k=10
    with torch.no_grad():
        masks = {"train":[]}
        
        re_train_loader , re_valid_loader , re_test_loader  = make_training_dataloaders(
            train_images=train_images,
            valid_images=valid_images,
            test_images=test_images,
            normalizer=normalizer,
            train_transforms=transforms["raw_"],
            test_transforms=transforms["raw_"],
            args=args
        )
        dataloaders = {
            "val" : [re_valid_loader,val_model],
            "test" : [re_test_loader,val_model]
        }

        for name in tqdm(train_names):
            label_name = name.split("_")[-1]
            mask = np.load(base_input_path / "train" / f"{label_name}.npy")
            mask[mask!=0]-=4
            mask[mask==20]-=4
            mask[mask==21]-=4

            pred_probs = np.load(train_folder/ "pred_probs" / f"{name}.npy")
            
            pred_mask = np.argmax(pred_probs,axis=0)
            pred_probs = pred_probs.astype(np.float16)
            pred_mask = pred_mask.astype(np.uint8)
            mask = mask.astype(np.uint8)

            masks["train"].append([mask,pred_mask,pred_probs,name,"raw_"])

        generate_dataset(
            masks = masks,
            lut = lut,
            root_idx_dict = root_locations,
            index_maps = index_maps,
            root_locations = root_locations,
            base_output_path = base_output_path,
            k=k,
            cnn_classes = []
        )
        del mask,re_train_loader
        gc.collect()

        masks = {"val":[],"test":[]}
        for part,(dataloader,model) in dataloaders.items():
            for data in tqdm(dataloader):
                """ 
                    mask : 0 to 18 classes
                    pred_mask : 0 to 6 classes
                """
                imgs = data["imgs"].cuda()
                mask = data["mask"]
                name = data["name"][0]
            
                with torch.autocast("cuda"):
                    pred_logits,aux = model(imgs)
                    pred_logits = pred_logits[0]
                    pred_probs = F.softmax(pred_logits,dim=1)[0]
                    pred_mask = torch.argmax(pred_logits,dim=1)[0]

                mask = mask.cpu().numpy()[0].astype(np.uint8)
                pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
                pred_probs = pred_probs.cpu().numpy().astype(np.float16)
                masks[part].append([mask,pred_mask,pred_probs,name,"raw_"])

                if(part == "val"):
                    for family in np.unique(pred_mask)[1:].tolist():
                        m = (pred_mask==family).astype(int)
                        selected_m = mask[m==1]
                        counts = Counter(selected_m.tolist())
                        for key,val in counts.items():
                            if(key == 0):
                                continue
                            co[family][old_class_map[key]]+=val            
        del re_test_loader , re_valid_loader
        del dataloaders
        del train_images , valid_images , test_images
        gc.collect()

    k = 14
    for family,values in co.items():
        top_k = heapq.nlargest(k, values.items(), key=lambda item: item[1])
        print(f"for family {family}")
        for key,val in top_k:
            print(f"{key} : {val}",end=" | ")
        print("\n")

    generate_dataset(
        masks = masks,
        lut = lut,
        root_idx_dict = root_locations,
        index_maps = index_maps,
        root_locations = root_locations,
        base_output_path = base_output_path,
        k=k,
        cnn_classes = []
    )
    del masks
    gc.collect()
    
    
    for part in ["train","val"]:
        names = os.listdir(base_output_path / part)
        gnn_preds,cnn_vessels,gt_vessels = load_set(
            names = names,
            base_gnn_path= base_output_path,
            part = part,
        )

        overall, per_class = evaluate_merged(
            gnn_preds = gnn_preds,
            cnn_vessels = cnn_vessels,
            gt_vessels = gt_vessels,
            family_labels = validator_class_maps,
            ignore_bg = False
        )

        print(overall)
        print(per_class)

        results = draw_gnn_masks(
            gnn_preds = gnn_preds,
            cnn_vessels = cnn_vessels,
            gt_vessels = gt_vessels,
            family_labels = validator_class_maps
        )

    os.makedirs(base_output_path / "gnn_mask_plots",exist_ok=True)
    for name, canvas in results.items():
        cv2.imwrite(base_output_path / "gnn_mask_plots" / f"{name}.png",canvas)

    with open(base_output_path / "dataset_keys.json" , "w") as f:
        json.dump(dataset_keys,f)
    with open(base_output_path / "class_maps.json" , "w") as f:
        json.dump(validator_class_maps,f)
    with open(base_output_path / "extra_index_indx.json","w") as f:
        json.dump(extra_index_indx,f)
    