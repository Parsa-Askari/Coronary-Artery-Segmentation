import torch
import numpy as np
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sknw
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
import torch.nn.functional as F
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import shutil
import albumentations as A
###IE###
from .draw_graph import draw_graph
from .gnn_helpers import (
    fix_16_family , find_right_root , 
    set_labels , find_dist_from_root , 
    sample_at_nodes , find_root,
    find_degree , find_radius,
    fit_feature_stats,load_feature_stats,
    apply_feature_stats,find_branch_depth,
    left_mapping_function,find_left_root,
    load_set,bridge_components,node_pixel_weights
)
from .gnn_coloring import compute_gnn_ceiling
from utils.gnn_evaluation import evaluate_merged
from utils.gnn_mask_drawing import draw_gnn_masks
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
        prob_feats,            # 9...:9+c per class model probs


    ]).astype(np.float32)
    

    return feats



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
        pred_mask,
        pred_probs,
        mask,
        root_idx_dict,
        index_maps,
        root_location,
        branch_index = -1,
        image_shape=512,
        k=10,
        base_graph_plot_path = None
    ):
    
    idx_to_keep = index_maps[branch_index]
    gt_mask = np.zeros_like(mask)
    for i,idx in enumerate(idx_to_keep) :
        gt_mask[mask==idx] = i+1
    

    binary_mask = (pred_mask==branch_index).astype(int).copy()
    binary_mask[-1,-1] = 1

    coords, ei , pred_skel = extract_graph(pred_mask=binary_mask,gt_mask=gt_mask,k=k)
    # root_idx = find_left_root(coords,pred_mask,branch_idx=1,
    #                             root_idx=root_idx_dict,axis=root_location)

    root_idx = find_left_root(coords = coords,pred_mask = pred_mask,
                                edge_index=ei,axis=root_location)
    features = create_features(
        coords,
        edge_index = ei,
        root_idx = root_idx,
        model_probs= pred_probs,
        binary_mask=binary_mask,
        image_shape=image_shape
    )
    node_labels = set_labels(coords,gt_mask)
    if(base_graph_plot_path is not None):
        draw_graph(
            coords=coords,
            edge_index=ei,
            node_labels=node_labels,
            skel= pred_skel,
            root_idx=root_idx,
            save_path = base_graph_plot_path
        )
    node_weights = node_pixel_weights(
        binary_mask, coords / image_shape,
        (image_shape, image_shape), power=0.5
    )
    res = {
        "features" : features,
        "ei" : ei,
        "node_labels" : node_labels,
        "binary_mask" : binary_mask,
        "pred_skel" : pred_skel,
        "new_gt_mask" : gt_mask,
        "node_weights" : node_weights
    }
    return res

def process_single_data(
    part,
    data, 
    lut, 
    root_idx_dict, 
    index_maps, 
    root_locations,
    base_output_path,
    k=1,
    cnn_classes = None
    ):

    mask, pred_mask,pred_probs, name,prefix = data

    if prefix == "raw_":
        prefix = ""
    

    current_name = f"{prefix}{name}" 
    base_edge_path = base_output_path / part / current_name / "edges"
    base_feature_path = base_output_path / part / current_name/ "features"
    base_label_path = base_output_path / part / current_name / "labels"
    base_pred_path = base_output_path / part / current_name / "pred_masks"
    base_true_path = base_output_path / part / current_name / "true_masks"
    base_weight_path = base_output_path / part / current_name / "node_weights"
    base_graph_plot_path = base_output_path / "graph_vis" / part

    os.makedirs(base_feature_path , exist_ok=True)
    os.makedirs(base_edge_path , exist_ok=True)
    os.makedirs(base_label_path , exist_ok=True)
    os.makedirs(base_pred_path , exist_ok=True)
    os.makedirs(base_true_path , exist_ok=True)
    os.makedirs(base_weight_path , exist_ok=True)

    gt_mask_classes = np.unique(lut[mask])[1:]
    pred_mask_classes = np.unique(pred_mask)[1:]

    model_missed_classes = np.setdiff1d(gt_mask_classes, pred_mask_classes)
    model_fp_classes = np.setdiff1d(pred_mask_classes,gt_mask_classes)
    model_correct_classes = np.intersect1d(gt_mask_classes, pred_mask_classes)

    ###########################
    #### Classes Captured
    ###########################
    for c in model_correct_classes.tolist():
        os.makedirs(base_graph_plot_path / str(c) , exist_ok=True)
        res = build_gnn_gt_dataset(
            pred_mask= pred_mask,
            pred_probs = pred_probs,
            mask=mask,
            k=k,
            branch_index=c,
            root_idx_dict=root_idx_dict,
            index_maps=index_maps,
            root_location=root_locations[c],
            base_graph_plot_path=base_graph_plot_path / str(c) / f"{name}.png"
        )
        if(cnn_classes is not None and c in cnn_classes):
            res["node_labels"] = np.ones_like(res["node_labels"]).astype(int)
        np.save(base_edge_path / f"{c}.npy", res["ei"])
        np.save(base_feature_path / f"{c}.npy", res["features"])
        np.save(base_label_path / f"{c}.npy", res["node_labels"])
        cv2.imwrite(base_pred_path / f"{c}.png", res["binary_mask"]*512)
        np.save(base_true_path / f"{c}.npy", res["new_gt_mask"])
        np.save(base_weight_path / f"{c}.npy", res["node_weights"])
    ###########################
    #### Classes Missed
    ###########################
    for c in model_missed_classes.tolist():
        idx_to_keep = index_maps[c]

        gt_mask = np.zeros_like(mask)
        for i, idx in enumerate(idx_to_keep):
            gt_mask[mask==idx] = i+1

        os.makedirs(base_graph_plot_path / str(c) , exist_ok=True)
        np.save(base_true_path / f"{c}.npy", gt_mask)

    ###########################
    #### Classes Overpredicted
    ###########################
    for c in model_fp_classes.tolist():
        os.makedirs(base_graph_plot_path / str(c) , exist_ok=True)
        res = build_gnn_gt_dataset(
            pred_mask=pred_mask,
            pred_probs = pred_probs,
            mask=mask,
            k=k,
            branch_index=c,
            root_idx_dict=root_idx_dict,
            index_maps=index_maps,
            root_location=root_locations[c],
            base_graph_plot_path=base_graph_plot_path / str(c) / f"{name}.png"
        )

        res["node_labels"] = np.ones_like(res["node_labels"]).astype(int)
        np.save(base_edge_path / f"{c}.npy", res["ei"])
        np.save(base_feature_path / f"{c}.npy", res["features"])
        np.save(base_label_path / f"{c}.npy", res["node_labels"])
        cv2.imwrite(base_pred_path / f"{c}.png", res["binary_mask"]*512)
        np.save(base_true_path / f"{c}.npy", res["new_gt_mask"])
        np.save(base_weight_path / f"{c}.npy", res["node_weights"])

def generate_dataset(
    masks,
    lut,
    root_idx_dict,
    index_maps,
    root_locations,
    base_output_path,
    k=1,
    cnn_classes = None
):
    tasks = []
    for part, dataset in masks.items():
        for data in dataset:
            tasks.append((part, data, lut, root_idx_dict, index_maps, root_locations,base_output_path,k,cnn_classes))
            # process_single_data(part, data, lut, root_idx_dict, index_maps, root_locations,base_output_path,k,cnn_classes)
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_single_data, *task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Generating Dataset"):
            pass

if __name__ == "__main__":
    old_class_map = {
        1: '5',2: '6',3: '7',4: '8',
        5: '9',6: '9a',7: '10',8: '10a',
        9: '11',10: '12',11: '12a',12: '13',
        13: '14',14: '14a',15: '15',
        16: '12b',17: '14b'
    }
    # validator_class_maps = {
    #     1:{0:"bg",1:"5",2:"12"},
    #     2:{0:"bg",1:"6",2:"9"}, #6,9
    #     3:{0:"bg",1:"7",2:"9a",3:"10"},#7,9a,10
    #     4:{0:"bg",1:"8",2:"10a"}, #8,10a
    #     5:{0:"bg",1:"11",2:"12a",3:"12b"},#11,12a,12b
    #     6:{0:"bg",1:"13",2:"14",3:"14a",4:"14b",5:"15"},#13,14,14a,14b,15
    # }
    # class_map = {
    #     1: '5', 2: '6', 3: '7', 4: '8', 5: '11', 6: '13'
    # }
    class_map = {
        1: 'LAD', 2: 'LCX', 3: '5' # LAD  , LCX , 5
    }
    root_locations = {
        1 : "left",
        2 : "left",
        3 : "topleft",
        4 : "topleft",
        5 : "topleft",
        6 : "topleft",
    }

    

    # index_maps = {
    #     1:[1,10], #5,12
    #     2:[2,5], #6,9
    #     3:[3,6,7],#7,9a,10
    #     4:[4,8], #8,10a
    #     5:[9,11,16],#11,12a,12b
    #     6:[12,13,14,17,15],#13,14,14a,14b,15
    # }
    # index_maps = {
    #     1:[1,10, 2,9], #5,12,6,11
    #     2:[2,5, 1,3,6], #6,9,5,7,9a
    #     3:[3,6,7, 4,2,4,8],#7,9a,10,6,8,10a
    #     4:[4,8, 3], #8,10a,7
    #     5:[9,11,16, 1,12,10,13],#11,12a,12b,5,13,12,14
    #     6:[12,13,14,17,15, 9,11,16],#13,14,14a,14b,15,11,12a,12b
    # }
    index_maps = {
        1:[2,3,4,5,6,7,8],
        2:[9,10,11,12,13,14,15,16,17],
        3:[1]
    }
    reversed_map = {v: k for k, v in class_map.items()}
    # root_idx_dict = {
    #     1:[1], # 5 : 5 itself is the root
    #     2:[1,2],#6 : connection point of 5 and 6 or highest point of 6
    #     3:[2,3],#7 : connection point of 6 and 7 or highest point of 7
    #     4:[3,4],#6 : connection point of 7 and 8 or highest point of 8
    #     5:[1,5],#6 : connection point of 11 and 5 or highest point of 11
    #     6:[5,6] #6 : connection point of 13 and 11 or highest point of 13
    # }
    root_idx_dict = {
        1:[1],
        2:[2,3,1],
        3:[3]
    }
    validator_class_maps = {}
    for i,ls in index_maps.items():
        validator_class_maps[i]={0:"bg"}
        for j,idx in enumerate(ls):
            validator_class_maps[i][j+1] = old_class_map[idx]


    lut = left_mapping_function(
        old_class_map = old_class_map,
        new_class_map = class_map
    )
    class_count = len(class_map) + 1
    #1,2,3,4,5,

    transforms = {
        "raw_":A.Compose([],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"}),
    }
    
    right = [1,2,3,4,20,21,22,23]

    base_input_path       = Path("./dataset/syntax/labels")
    base_output_path      = Path(f"./dataset/syntax/gnn_dataset/gt_left_dataset_{len(class_map)}")

    masks = {"train":[],"val":[],"test":[]}

    if base_output_path.exists() and base_output_path.is_dir():
        shutil.rmtree(base_output_path)

    for part in ["train","val","test"]:
        names = os.listdir(base_input_path / part)
        for name in tqdm(names):

            name_stem = Path(name).stem
            mask = np.load(base_input_path / part / name)

            us = np.unique(mask)
            u = int(us[us!=0][0])
            if(u in right):
                continue

            mask[mask!=0]-=4
            mask[mask==20]-=4
            mask[mask==21]-=4

            pred_mask = lut[mask].copy()

            masks[part].append([mask,pred_mask,np.zeros((len(class_map)+1,pred_mask.shape[0],pred_mask.shape[1])),name_stem])
            
    generate_dataset(
        masks = masks,
        transforms = transforms,
        lut = lut,
        root_idx_dict = root_idx_dict,
        index_maps = index_maps,
        root_locations = root_locations,
        base_output_path=base_output_path
    )

    part = "train"
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
        family_labels = validator_class_maps
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


    dataset_keys = {
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
    base_key_size= len(dataset_keys) 
    for i in range(len(class_map)+1):
        dataset_keys[f"prob_{i}"]=i+base_key_size

    with open(base_output_path / "dataset_keys.json" , "w") as f:
        json.dump(dataset_keys,f)
    with open(base_output_path / "class_maps.json" , "w") as f:
        json.dump(validator_class_maps,f)