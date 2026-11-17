import torch
import numpy as np
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sknw
import json
from draw_graph import draw_graph
import networkx as nx
import torch.nn.functional as F
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import shutil
import albumentations as A
###IE###
from gnn_helpers import (
    fix_16_family , find_right_root , 
    set_labels , find_dist_from_root , 
    sample_at_nodes , find_root,
    find_degree , find_radius,
    fit_feature_stats,load_feature_stats,
    apply_feature_stats,find_branch_depth
)
from gnn_coloring import compute_gnn_ceiling
###SS###
NB = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]  # 8-connectivity

def create_features(
        coords, 
        edge_index, 
        root_idx,
        model_probs,
        binary_mask, 
        image_shape=512
    ):
    N = coords.shape[0]
    C = model_probs.shape[0]

    norm_coords = coords.astype(np.float32) / image_shape  

    geo_n , euc_n , man_n , reachable = find_dist_from_root(
        root_idx = root_idx,
        edge_index = edge_index,
        coords = coords,
        image_shape = image_shape,
        N = N
    )

    degree = find_degree(edge_index, N)
    radius = find_radius(binary_mask, coords)

    prob_feats = np.stack(
        [sample_at_nodes(model_probs[c], coords) for c in range(C)],
        axis=1,
    ) 
    branch_depth, _ = find_branch_depth(coords, edge_index, root_idx, N)

    feats = np.column_stack([
        norm_coords,           # 0,1: x, y
        geo_n,                 # 2: geodesic hop distance (normalised, -1 if unreachable)
        euc_n,                 # 3  euclidean distance to root
        man_n,                 # 4:  manhattan distance to root
        reachable,             # 5:  1 if geodesic valid, 0 if disconnected from root
        degree ,               # 6: degree of the node
        radius ,               # 7: the radius of the vessel
        branch_depth ,         # 8: the branch number the node belongs to
        prob_feats,            # 8...:8+c per class model probs


    ]).astype(np.float32)
    keys = {
        "x":0,
        "y":1,
        "geo_root":2,
        "euc_root":3,
        "man_root":4,
        "reachable":5,
        "degree":6,
        "radius":7,
        "branch_depth":8
    }
    base_key_size= len(keys) 
    for i in range(C):
        keys[f"prob_{i}"]=i+base_key_size

    return feats,keys



def make_graph(skel, k=1):

    G = sknw.build_sknw(np.ascontiguousarray(skel).astype(np.uint16))
 
    coords = []   
    key_id = {}       
    for nid in G.nodes():
        key_id[nid] = len(coords)
        coords.append(G.nodes[nid]["o"])  
 
    edges = []
    for a, b in G.edges():
        pts = G[a][b]["pts"]             
        chain = [key_id[a]]      
 
        if k > 1:
            for i in range(k, len(pts), k):
                if i >= len(pts) - 1: 
                    break
                chain.append(len(coords))
                coords.append(pts[i].astype(float))
 
        chain.append(key_id[b])
        for u, v in zip(chain[:-1], chain[1:]):
            edges.append((u, v))
 
    coords = np.asarray(coords, dtype=np.float32)
    if edges:
        ei = np.array(edges, dtype=np.int64).T
    else:
        ei = np.zeros((2, 0), dtype=np.int64)

    ei = np.concatenate([ei, ei[::-1]], axis=1) 
    return coords, ei

def make_graph_dense(skel):

    Xs, Ys = np.where(skel > 0)
    coords = np.stack([Xs, Ys], 1).astype(np.float32)
    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(Xs, Ys))}
    edges = []
    for (x, y), i in idx.items():
        for dx, dy in NB:
            j = idx.get((x + dx, y + dy))
            if j is not None and i < j: 
                edges.append((i, j))

    ei = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), np.int64)
    ei = np.concatenate([ei, ei[::-1]], axis=1) 
    return coords, ei
        
def extract_graph(pred_mask,gt_mask,k=1):
    """
        pred_mask : (H,W)
        gt_mask : (H,W)
        side : both , right , left
    """
    if(isinstance(gt_mask,torch.Tensor)):
        gt_mask = torch.from_numpy(gt_mask)
    if(isinstance(pred_mask,torch.Tensor)):
        pred_mask = torch.from_numpy(pred_mask.cpu().numpy())

    
    pred_mask = (pred_mask!=0).astype(int)
    pred_skel = skeletonize(image=pred_mask,method="lee")
    if(k==1):
    
        coords, ei = make_graph_dense(pred_skel)
    else:
        coords, ei = make_graph(pred_skel,k=k)
    return coords, ei , pred_skel

def build_gnn_gt_dataset(
        base_input_path,
        base_output_path,
        augment_ds_path_base,
        part,
        t=None,
        image_shape=512,
        k=10,
        side="both",
        small_classes_only = True,
        device = "cuda",
        index_16 = 20,
        prefix = ""
    ):
    base_input_path = Path(base_input_path)
    base_output_path = Path(base_output_path)
    augment_ds_path_base = Path(augment_ds_path_base)
    
    
    base_edge_path = base_output_path / part / "edges" 
    base_feature_path = base_output_path / part / "features"
    base_label_path = base_output_path / part / "labels" 
    base_graph_plot_path = base_output_path / part / "graph_vis" 

    base_gt_mask_path = base_output_path / "model_preds" / part 

    base_aug_masks = augment_ds_path_base / "labels" / part
    base_aug_imgs = augment_ds_path_base / "images" / part

    os.makedirs(base_edge_path,exist_ok=True)
    os.makedirs(base_feature_path,exist_ok=True)
    os.makedirs(base_label_path,exist_ok=True)
    os.makedirs(base_graph_plot_path,exist_ok=True)
    os.makedirs(base_gt_mask_path,exist_ok=True)
    os.makedirs(base_aug_masks,exist_ok=True)
    os.makedirs(base_aug_imgs,exist_ok=True)

    mask_names = os.listdir(base_input_path / part)
    mask_names = [n for n in mask_names if ".npy" in n]
    i=0
    for name in tqdm(mask_names):
        name_stem = Path(name).stem
        gt_mask = np.load(base_input_path / part / name)
        if(t is not None):
            r = t(image = np.ones((512, 512)),mask=gt_mask)
            gt_mask = r["mask"]

        if(side == "right"):
            if(small_classes_only):
                if(1 not in gt_mask or 20 not in gt_mask):
                    i +=1
                    continue
                root_class_mask = (gt_mask == 3)
                gt_mask = fix_16_family(gt_mask,index_16=index_16)
            else:
                if(1 not in gt_mask):
                    continue

                root_class_mask = (gt_mask==1)
                gt_mask[gt_mask>4]-= 15
            
    
        binary_mask = gt_mask.copy()
        binary_mask = (binary_mask!=0).astype(int)
        binary_mask[0,0]=1

        coords, ei , pred_skel = extract_graph(pred_mask=binary_mask,gt_mask=gt_mask,k=k)
        root_idx = find_root(coords,root_class_mask)

    
        features,dataset_keys = create_features(
            coords,
            edge_index = ei,
            root_idx = root_idx,
            model_probs= np.ones((2,512,512),dtype=np.float32),
            binary_mask=binary_mask,
            image_shape=image_shape
        )
        node_labels = set_labels(coords,gt_mask)
        
        
        draw_graph(
            coords=coords,
            edge_index=ei,
            node_labels=node_labels,
            skel= pred_skel,
            root_idx=root_idx,
            save_path = base_graph_plot_path / f"{name_stem}.png"
        )

        np.save(base_edge_path / f"{prefix}{name}", ei)
        np.save(base_feature_path / f"{prefix}{name}", features)
        np.save(base_label_path / f"{prefix}{name}", node_labels)
        cv2.imwrite(base_gt_mask_path/ f"{prefix}{name_stem}.png",binary_mask*512)
        np.save(base_aug_masks / f"{prefix}{name}" , gt_mask)

    with open(base_output_path / "dataset_keys.json","w") as f:
        json.dump(dataset_keys,f)

@torch.no_grad()
def build_model_dataset(
        model,
        dataloader,
        base_output_path,
        base_output_aug_masks,
        part,
        root_model = None,
        root_c = 3,
        image_shape=512,
        k=10,
        side="both",
        small_classes_only = True,
        device = "cuda",
        index_16 = 20,
    ):

    base_output_path = Path(base_output_path)
    base_output_aug_masks = Path(base_output_aug_masks)



    
    augment_ds_path_base = base_output_aug_masks / "gnn_dataset_augmented_images" 
    if (augment_ds_path_base/part).exists() and (augment_ds_path_base/part).is_dir():
            shutil.rmtree(augment_ds_path_base/part)
   
    base_edge_path = base_output_path / part / "edges" 
    base_feature_path = base_output_path / part / "features"
    base_label_path = base_output_path / part / "labels" 
    base_graph_plot_path = base_output_path / part / "graph_vis" 

    base_gt_mask_path = base_output_path / "model_preds" / part 

    base_aug_masks = base_output_aug_masks / "gnn_dataset_augmented_images" / "labels" / part
    base_aug_imgs = base_output_aug_masks / "gnn_dataset_augmented_images" / "images" / part

    
    

    os.makedirs(base_edge_path,exist_ok=True)
    os.makedirs(base_feature_path,exist_ok=True)
    os.makedirs(base_label_path,exist_ok=True)
    os.makedirs(base_graph_plot_path,exist_ok=True)
    os.makedirs(base_gt_mask_path,exist_ok=True)
    os.makedirs(base_aug_masks,exist_ok=True)
    os.makedirs(base_aug_imgs,exist_ok=True)

    
    for data in tqdm(dataloader):
        img_base = data["new_image_not_normal"]
        img = data["new_image"].to(device)
        gt_mask = data["mask"][0].cpu().numpy()
        name = data["name"][0]
        name_stem = Path(name).stem

        if(side == "right"):
            if(small_classes_only):
                if(1 not in gt_mask or 5 not in gt_mask):
                    continue
                root_class_mask = (gt_mask == 3)
                gt_mask = fix_16_family(gt_mask,index_16=index_16)
                
            

        with torch.autocast(device_type=device):
            pred_logits = model(img)[0]
            per_class = F.softmax(pred_logits, dim=1)[0] 
            per_class = per_class.detach().cpu().numpy()

            pred_mask = torch.argmax(pred_logits,dim=1).cpu().numpy()[0]
            if(side == "right"):
                if(small_classes_only):
                    if(5 not in pred_mask):
                        continue
                    pred_mask[pred_mask!=5] = 0
                    pred_mask[pred_mask==5] = 1
                else:
                    root_class_mask = np.logical_and(gt_mask==1,pred_mask)
                    
            binary_mask = (pred_mask!=0).astype(int)
                
        coords, ei , pred_skel = extract_graph(pred_mask=binary_mask,gt_mask=gt_mask,k=k)
        root_idx = find_root(coords,root_class_mask)
        # root_idx = find_right_root(coords, root_class_mask,c=root_c) 
        node_labels = set_labels(coords,gt_mask)

        
        features , dataset_keys= create_features(
            coords,
            edge_index = ei,
            root_idx = root_idx,
            model_probs = per_class,
            image_shape=image_shape,
            binary_mask = binary_mask
        )
        
            
        
        draw_graph(
            coords=coords,
            edge_index=ei,
            node_labels=node_labels,
            skel= pred_skel,
            root_idx=root_idx,
            save_path = base_graph_plot_path / f"{name_stem}.png"
        )
        
        np.save(base_edge_path / name, ei)
        np.save(base_feature_path / name, features)
        np.save(base_label_path / name, node_labels)
        cv2.imwrite(base_gt_mask_path/ f"{name_stem}.png",binary_mask*512)

        np.save(base_aug_masks / name , gt_mask)
    with open(base_output_path / "dataset_keys.json","w") as f:
        json.dump(dataset_keys,f)
if __name__ == "__main__":
    # pred_path = "./dataset/syntax/labels/train/97.npy"
    # gt_path = "./dataset/syntax/labels/train/97.npy"
    
    # pred = np.load(pred_path)
    # gt = np.load(gt_path)

    # pred[pred<20] = 0
    # gt[gt<20] = 0

    # coords, ei , pred_skel = extract_graph(pred_mask=pred,gt_mask=gt,k=10)
    # features = create_features(coords,image_shape=512)

    # rr = np.clip(coords[:,0].round().astype(int), 0, pred.shape[0]-1)
    # cc = np.clip(coords[:,1].round().astype(int), 0, pred.shape[1]-1)
    # node_labels = gt[rr, cc]
    # draw_graph(coords=coords,edge_index=ei,node_labels=node_labels,skel= pred_skel)
    branch = 1
    transforms = {
        "raw_":A.Compose([],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"}),
        # "aff_":A.Compose([A.Affine(
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
    cols = [2, 3, 4, 5]

    base_input_path       = "./dataset/syntax/new_labels"
    base_output_path      = "./dataset/syntax/gnn_dataset/k_gt_dataset"
    base_output_aug_masks = Path("./dataset/syntax/gnn_dataset_augmented_images")

    if (Path(base_output_path)).exists() and (Path(base_output_path)).is_dir():
            shutil.rmtree(base_output_path)
    if base_output_aug_masks.exists() and base_output_aug_masks.is_dir():
                    shutil.rmtree(base_output_aug_masks)

    os.makedirs(base_output_aug_masks,exist_ok=True)
    for prefix,t in transforms.items():
        build_gnn_gt_dataset(
            base_input_path=base_input_path,
            base_output_path=base_output_path,
            augment_ds_path_base=base_output_aug_masks,
            side="right", k=1,
            small_classes_only=True, index_16=5,
            prefix = prefix,
            t = t,
            part="train"
        )
    for part in ["val","test"]:
        build_gnn_gt_dataset(
            base_input_path=base_input_path,
            base_output_path=base_output_path,
            augment_ds_path_base=base_output_aug_masks,
            side="right", k=1,
            small_classes_only=True, index_16=5,
            prefix = "",
            t = None,
            part=part
        )
    
    compute_gnn_ceiling(

        gt_root = Path(base_output_aug_masks) / "labels",
        pred_root = Path(base_output_path) / "model_preds",
        gnn_root = base_output_path,
        # class_map = {0:"bg",1:"1",2:"2",3:"3",4:"4",5:"16",6:"16a",7:"16b",8:"16c"}
        class_map = {0:"bg",1:"16",2:"16a",3:"16b",4:"16c"}
    )
    out = Path(base_output_path)  
    stats_path = out / "stats.npz"



    # print("starting to normalize")
    # for part in ["train", "test", "val"]:
    #     feat_dir = out / part / "features"
    #     for name in os.listdir(feat_dir):
    #         f = feat_dir / name
    #         features = np.load(f)
    #         features[:,cols] = apply_feature_stats(
    #             features[:,cols], 
    #             mean[cols], 
    #             std[cols]
    #         )
    #         np.save(f, features) 