import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import albumentations as A
import cv2
from pathlib import Path
import shutil
###IE###
from models.swin_upernet_cascade import SwinUperNetCascade
from models.smp_models import DeepLabV3Plus
from models.S3_DSCNet_pro import DSCNet_pro
from utils.helpers import load_model,read_images,make_result_dataloaders
from utils.gnn_dataset_generation_args import args
from utils.graph_extraction import (
    generate_robust_mask,
    create_gnn_dataset_parallel_limited
)
from utils.k_step_graph_extraction import build_model_dataset
from utils.gnn_coloring import compute_gnn_ceiling
###SS###
class EnsembledModels(torch.nn.Module):
    def __init__(self, paths):
        super().__init__()
        self.models = torch.nn.ModuleList([load_model(model_class= c, path=p,device="cuda",model_inputs={}) for p,c in paths])
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def forward(self, img, side=None):
        probs = torch.stack([
            F.softmax(m(img, side)[0].float(), dim=1) for m in self.models
        ]).mean(0)
        # return LOG-probs: argmax is unchanged, and a downstream softmax is a
        # no-op that recovers the averaged probs. Drop-in for your eval code.
        return [torch.log(probs.clamp_min(1e-8))]

""" 
valid_paths = [
        ["./outputs/2026-07-16_17-29-02 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-17_00-28-41 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-17_01-04-15 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        # ["./outputs/2026-07-17_10-41-50 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-19_14-27-22 [model[DeepLabV3Plus]-multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",DeepLabV3Plus]
    ]
train_paths = {
        # "ep[5]_model[swin]":["./outputs/2026-07-19_16-07-10 [model[SwinUperNetCascade]-multi-right-6class-5epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        "ep[10]_model[swin]":["./outputs/2026-07-19_15-51-19 [model[SwinUperNetCascade]-multi-right-6class-10epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        # "ep[15]_model[swin]":["./outputs/2026-07-19_16-07-10 [model[SwinUperNetCascade]-multi-right-6class-5epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        # "ep[50]_model[deeplab]":["./outputs/2026-07-19_14-27-22 [model[DeepLabV3Plus]-multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",DeepLabV3Plus],
    }
"""
## binary model
"""
train_paths = {
    "ep[50]_model[snake]":["./outputs/2026-07-22_15-07-04 [model[DSCNet_pro]-binary-right-2class-50epo-aux_branchpoint-nobias-cbam-upsample]",DSCNet_pro]
}
valid_paths = [
    ["./outputs/2026-07-22_15-07-04 [model[DSCNet_pro]-binary-right-2class-50epo-aux_branchpoint-nobias-cbam-upsample]",DSCNet_pro]
]
"""
if __name__ == "__main__":
    
    valid_paths = [
        ["./outputs/2026-07-16_17-29-02 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-17_00-28-41 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-17_01-04-15 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        # ["./outputs/2026-07-17_10-41-50 [multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        ["./outputs/2026-07-19_14-27-22 [model[DeepLabV3Plus]-multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",DeepLabV3Plus]
    ]
    train_paths = {
        # "ep[5]_model[swin]":["./outputs/2026-07-19_16-07-10 [model[SwinUperNetCascade]-multi-right-6class-5epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        "ep[10]_model[swin]":["./outputs/2026-07-19_15-51-19 [model[SwinUperNetCascade]-multi-right-6class-10epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        # "ep[15]_model[swin]":["./outputs/2026-07-19_16-07-10 [model[SwinUperNetCascade]-multi-right-6class-5epo-aux_branchpoint-nobias-cbam-upsample]",SwinUperNetCascade],
        # "ep[50]_model[deeplab]":["./outputs/2026-07-19_14-27-22 [model[DeepLabV3Plus]-multi-right-6class-50epo-aux_branchpoint-nobias-cbam-upsample]",DeepLabV3Plus],
    }
    
    normalizer = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
    )

    transforms = {
        "raw":A.Compose([],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"}),
        # "aff":A.Compose([A.Affine(
        #         translate_percent=(-0.10, 0.10),
        #         scale=(0.9, 1.1),       
        #         rotate=(-30, 30),           
        #         border_mode=cv2.BORDER_CONSTANT,
        #         fill=0,
        #         fill_mask=0,
        #         p=1
        #     )],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"}),
        # "elas":A.Compose([A.ElasticTransform(
        #         alpha=300.0,
        #         sigma=8.0,         
        #         p=1.0
        #     )],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"}),
        # "grid":A.Compose([A.GridDistortion(
        #         num_steps=5, 
        #         distort_limit=(-0.5,0.5),  
        #         p=1.0
        #     )],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"})
    }
    
    
    train_images = read_images(
        paths = args["paths"],
        part = "train",
        chosen_labels=None,
        hard_classes = None,
        bootstrap = args["bootstrap"],
        training_mode = args["training_mode"],
        binary_mode = args["binary_training_mode"],
        side_mode = args["side_mode"],
        in_c = args["in_c"],
        cut_borders = args["cut_borders"],
        class_count = args["class_count"]

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
        class_count = args["class_count"]
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
        class_count = args["class_count"]
    )

    base_output_path = "./dataset/syntax/gnn_dataset/k_step_dataset"
    base_output_aug_masks = "./dataset/syntax"
    if (Path(base_output_path)).exists() and (Path(base_output_path)).is_dir():
        shutil.rmtree(base_output_path)
    ## generate train set
    for model_name , model_data in train_paths.items():
        for aug_name , transform in transforms.items():
            print(f"generating {model_name}-{aug_name}")

            re_train_loader , re_valid_loader , re_test_loader  = make_result_dataloaders(
                train_images=train_images,
                valid_images=valid_images,
                test_images=test_images,
                normalizer=normalizer,
                transforms=transform
            )
            dataloaders = {
                "train":re_train_loader,
                # "val":re_valid_loader,
                # "test":re_test_loader
            }
            
            model = load_model(
                model_class=model_data[1],
                path=model_data[0],
                model_inputs={},
                device="cuda",
                return_args = False
            )
            model.eval()
            for part , dataloader in dataloaders.items():
                build_model_dataset(
                    model = model ,
                    dataloader= dataloader,
                    base_output_path=base_output_path,
                    base_output_aug_masks = base_output_aug_masks,
                    root_c=1,
                    part = part,
                    k=1,
                    image_shape=512,
                    index_16 = 5,
                    side="right",
                    small_classes_only=True,
                )
            
            del model

    ### valid dataset
    
    print(f"generating valid set")
    re_train_loader , re_valid_loader , re_test_loader  = make_result_dataloaders(
        train_images=train_images,
        valid_images=valid_images,
        test_images=test_images,
        normalizer=normalizer,
        transforms=transforms["raw"]
    )
    dataloaders = {
        # "train":re_train_loader,
        "val":re_valid_loader,
        "test":re_test_loader
    }
    valid_model = EnsembledModels(valid_paths)
    valid_model.eval()
    for part , dataloader in dataloaders.items():
        build_model_dataset(
            model = valid_model ,
            dataloader= dataloader,
            base_output_path=base_output_path,
            base_output_aug_masks = base_output_aug_masks,
            root_c=1,
            part = part,
            k=1,
            image_shape=512,
            side="right",
            index_16 = 5,
            small_classes_only=True,
        )

    compute_gnn_ceiling(

        gt_root = Path(base_output_aug_masks) / "gnn_dataset_augmented_images" / "labels",
        pred_root = Path(base_output_path) / "model_preds",
        gnn_root = base_output_path,
        class_map = {0:"bg",1:"16",2:"16a",3:"16b",4:"16c"}
    )
