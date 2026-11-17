import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from skimage.filters import apply_hysteresis_threshold
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from skimage.morphology import skeletonize
import networkx as nx
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import shutil
from IPython.display import display
import zarr
###IE###
from .helpers import (
    make_skel_from_features,
    calaculate_gnn_multi_dice,
    calaculate_gnn_binary_dice,
    draw_mask,draw_fast_plots
)
###SS###
NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
               ( 0, -1),          ( 0, 1),
               ( 1, -1), ( 1, 0), ( 1, 1)]  
def find_root_id(graph,dist_transform,nodes,image_shape):
    candidates = []
    for node_id in graph.nodes():
        if(graph.degree[node_id]==1):
            candidates.append(node_id)
            
    if(len(candidates)==0):
        candidates = list(graph.nodes())

    H, W = image_shape
    best_root = None
    best_score = -float('inf')

    max_r = max([dist_transform[nodes[n][0], nodes[n][1]] for n in candidates]) if candidates else 1.0
    if max_r == 0: 
        max_r = 1.0

    for node_id in candidates:
        x,y = nodes[node_id]
        r = dist_transform[x,y]
        score_radius = r / max_r
        
        score_height = 1.0 - (x / H)

        dist_to_edge = min(x, H - x, y, W - y)
        score_edge = 1.0 - (dist_to_edge / (min(H, W) / 2))

        total_score = (1.5 * score_radius) + (2.0 * score_height) + (0.5 * score_edge)

        if total_score > best_score:
            best_score = total_score
            best_root = node_id
    
    
    return best_root

def get_dist_from_branchpoint(G):
    degrees = dict(G.degree())
    branchpoints = [
        nid for nid, deg in degrees.items()
        if deg >= 3
    ]
    if len(branchpoints) == 0:
        return {nid : -1 for nid in degrees},{nid:[-1] for nid in degrees}
    dist, paths = nx.multi_source_dijkstra(
        G,
        sources=branchpoints,
        weight="weight"
    )
    return dist , paths
def get_dist_from_endpoint(G):
    endpoints = [n for n in G.nodes if G.degree[n] == 1]

    if len(endpoints) == 0:
        return {n: 0.0 for n in G.nodes}

    dist = nx.multi_source_dijkstra_path_length(
        G,
        sources=endpoints,
        weight="weight"
    )
    nodes = list(G.nodes)
    dist_endpoint_feature = {n:dist.get(n, 0.0) for n in nodes}
        
    return dist_endpoint_feature

def get_positional_root_dist(G,nodes):
    components = nx.connected_components(G)
    positional_dist = {}
    for comp_id , comp_nodes in enumerate(components):
        subG = G.subgraph(comp_nodes).copy()
        degrees = dict(subG.degree())
        endpoints = [i for i in degrees if degrees[i]==1]
        
        if(len(endpoints)==0):
            endpoints = list(subG.nodes)

        root = min(endpoints,key = lambda n_id : nodes[n_id][0]+nodes[n_id][1])
        dist = nx.single_source_dijkstra_path_length(
            subG,root,weight="weight"
        )
        max_dist = max(dist.values()) if len(dist)!=0 else 1.0
        if max_dist == 0:
            max_dist = 1.0
        for n in subG.nodes : 
            positional_dist[n] = dist[n]/max_dist
    return positional_dist

def feature_extraction(binary_mask,prob_mask,img_base,frangi_mask,
                       nodes,edges,labels,path):
    G = nx.Graph()
    G.add_nodes_from(nodes.keys())
    for edge_id , edge_nodes in edges.items():
        a,b  = edge_nodes
        ra, ca = nodes[a]
        rb, cb = nodes[b]
        w = np.sqrt((ra - rb) ** 2 + (ca - cb) ** 2)
        G.add_edge(edge_nodes[0],edge_nodes[1],weight=w)
    n_nodes = len(nodes)

    b_dists, b_paths = get_dist_from_branchpoint(G)
    e_dists = get_dist_from_endpoint(G)
    positional_dist = get_positional_root_dist(G,nodes=nodes)

    dist_from_bg = cv2.distanceTransform(binary_mask,cv2.DIST_L2,5)
    
    dist_from_root = {}
    if(len(G)>0):
        largest_cc = max(nx.connected_components(G),key=len)
        largest_g = G.subgraph(largest_cc).copy()
        root_node_id = find_root_id(
            graph = largest_g, 
            dist_transform = dist_from_bg, 
            nodes = nodes,
            image_shape = binary_mask.shape
        )

        dist_from_root = nx.single_source_shortest_path_length(
            largest_g, 
            root_node_id
        )
    
    # largest_g = G.subgraph(largest_cc).copy()

    
    # root=find_root_id(graph=largest_g,dist_transform=dist_from_bg,nodes=nodes,image_shape=binary_mask.shape)
    # rgb = (cv2.cvtColor(binary_mask,cv2.COLOR_GRAY2RGB)*255).astype(np.uint8)
    # x,y = nodes[root]
    # cv2.circle(rgb, (y, x), 5, (0, 0, 255), -1)
    # cv2.imwrite(path,rgb)
    comp_size_dict = {}
    for comp in nx.connected_components(G):
        size = len(comp)
        for n in comp:
            comp_size_dict[n] = size
    degrees = dict(G.degree())

    gts = np.zeros(len(labels), dtype=np.int64)
    features = np.zeros((len(nodes),16))

    max_bg_dist = dist_from_bg.max()

    max_dim = max(binary_mask.shape[0]-1,binary_mask.shape[1]-1)
    max_branchpoint_dist = abs(max(b_dists.values())) if len(b_dists) > 0 else 1.0
    max_endpoint_dist = max(e_dists.values()) if len(e_dists) > 0 else 1.0
    max_root_dist = max(dist_from_root.values()) if len(dist_from_root) > 0 else 1.0
    
    if max_branchpoint_dist == 0:
        max_branchpoint_dist = 1.0
    if max_endpoint_dist == 0:
        max_endpoint_dist = 1.0
    if max_root_dist == 0:
        max_root_dist = 1.0

    img0 = img_base[0]

    for node_id,cords in nodes.items():
        ### cord calc
        x,y = cords
        ### degree calc
        degree = degrees[node_id]
        ### redius calc
        redius = dist_from_bg[x,y]
        ### comp size calc
        comp_size = comp_size_dict[node_id]
        ### Root dist calc
        dist_to_root = dist_from_root.get(node_id, -1)
        ### intensity calc
        intensity = img0[x, y]
        ### prob calc
        prob = prob_mask[x,y]
        ### frangi filter
        f_score = frangi_mask[x,y]
        ### endpoint and brachpoint 
        is_endpoint = 1 if degree==1 else 0
        is_brachpoint = 1 if degree>=3 else 0
        ### dist from brachpoints
        path = b_paths.get(node_id,[-1])
        nearest_bp = path[0]
        distance_to_branchpoint = b_dists.get(node_id,-1)
        nearest_branchpoint_id = nearest_bp
        has_brachpoint = 1 if nearest_bp !=-1 else 0
        ### downstream pixels
        distance_to_endpoint = e_dists[node_id]
        ### positional dist from root
        node_positional_dist= positional_dist[node_id]
        ###### Normalization
        x = x / max_dim
        y = y / max_dim
        degree = degree / 8.0
        redius = redius / (max_bg_dist + 1e-6)
        comp_size = comp_size / n_nodes

        if dist_to_root != -1:
            dist_to_root = dist_to_root / (max_root_dist + 1e-6)
        else:
            dist_to_root = -1

        intensity = intensity / 255.0
        distance_to_branchpoint = distance_to_branchpoint / max_branchpoint_dist
        distance_to_endpoint = distance_to_endpoint / (max_endpoint_dist + 1e-6)

        ##################################
        features[node_id] = [
            x,
            y,
            degree,
            redius,
            comp_size,
            dist_to_root,
            intensity,
            prob,
            f_score,
            is_endpoint,
            is_brachpoint,
            distance_to_branchpoint,
            nearest_branchpoint_id,
            has_brachpoint,
            distance_to_endpoint,
            node_positional_dist
        ]

        label = labels[node_id]
        gts[node_id]= int(label)
    return features,gts

def hysteresis_threshold(probability_map, low_thresh=0.2, high_thresh=0.7):
    binary_mask = apply_hysteresis_threshold(
        probability_map, 
        low=low_thresh, 
        high=high_thresh
    )
    binary_mask = binary_mask.astype(np.uint8)
    return binary_mask

@torch.no_grad()
def generate_helper(model,dataloader,output_base_path,low_thresh=0.2, high_thresh=0.7,save_mode="numpy"):
    for img_base , img , mask , side,name  in tqdm(dataloader):
        img = img.to("cuda")
        name = name[0]
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            pred_masks = model(img)[0]
            pred_probs = F.softmax(pred_masks,dim=1)[0,1].cpu().numpy()

        corrected_mask = hysteresis_threshold(pred_probs,low_thresh, high_thresh)
        pred_mask = (pred_probs>=0.5).astype(np.uint8)
        binray_mask = (mask[0]!=0).int()
        binray_mask = (binray_mask.numpy()*255).astype(np.uint8)
        if(save_mode=="plot"):
            fig = plt.figure(figsize=(15,5))
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(corrected_mask,cmap="gray")
            ax1.set_title("corrected")
            ax1.axis("off")

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(pred_mask,cmap="gray")
            ax2.set_title("oroginal")
            ax2.axis("off")
            
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(binray_mask,cmap="gray")
            ax3.set_title("gt")
            ax3.axis("off")

            fig.savefig(os.path.join(output_base_path, f"{name}.png"), 
                        bbox_inches="tight", dpi=200)
            plt.close(fig)
        else:
            np.save(os.path.join(output_base_path, f"{name}.npy"),corrected_mask)

@torch.no_grad()
def generate_robust_mask(model,train_loader,valid_loader,low_thresh=0.2, high_thresh=0.7,
                         output_path = "./dataset/syntax/fixed_binary_masks",save_mode="numpy"):
    train_base_path = os.path.join(output_path,"train")
    valid_base_path = os.path.join(output_path,"val")

    os.makedirs(output_path,exist_ok=True)
    os.makedirs(train_base_path,exist_ok=True)
    os.makedirs(valid_base_path,exist_ok=True)

    generate_helper(model,train_loader,train_base_path,
                    low_thresh=low_thresh, high_thresh=high_thresh,save_mode=save_mode)
    generate_helper(model,valid_loader,valid_base_path,
                    low_thresh=low_thresh, high_thresh=high_thresh,save_mode=save_mode)

def find_neighbors(mask_skel,cords):
    """
    Inputs : 
        - mask_skel: (H,W)
        - param coreds: (x,y)
    """
    neighbors = []
    H,W = mask_skel.shape
    for off_x , off_y in NEIGHBORS_8:
        x_ , y_ = cords[0]+off_x , cords[1]+off_y
        if(x_ >= H or x_< 0):
            continue
        if(y_ >= W or y_ < 0):
            continue
        if(mask_skel[x_,y_]==1):
            neighbors.append((x_,y_))
    return neighbors

def build_edges(nodes, nodes_to_id, mask_skel):
    edge_list = []
    
    for curr_id, node in nodes.items():
        neighbors = find_neighbors(mask_skel, node)
        for neighbor in neighbors:
            ne_id = nodes_to_id[neighbor]
            if curr_id < ne_id:
                edge_list.append([curr_id, ne_id])
                
    edges = {i: edge_list[i] for i in range(len(edge_list))}
    return edges

def build_nodes(Xs,Ys):
    nodes = {i:(Xs[i],Ys[i]) for i in range(len(Xs))}
    nodes_to_id = {(Xs[i],Ys[i]):i for i in range(len(Xs))}
    return nodes,nodes_to_id

def build_graph(nodes,edges):
    graph = nx.Graph()
    for node_id , node in nodes.items():
        graph.add_node(node_id,pt=node)
    for edge_id , edge in edges.items():
        graph.add_edge(edge[0],edge[1],edge_id=edge_id)
    return graph

def labeling_nodes(nodes,gt_mask):
    node_ids_to_classes = {}
    for node_id,node in nodes.items() : 
        x,y = node
        label = gt_mask[x,y]
        node_ids_to_classes[node_id] = label
    return node_ids_to_classes
def binary_kernel_based_labeling_nodes(nodes,gt_mask,r=2,min_votes=1):
    """
        - nodes : node_id -> (x,y)
        - gt_mask : (H,W) [0-26]
        - th_when_fg : if gt == 1 and vote >= th_when_fg then 1
        - th_when_bg : if gt == 0 and vote >= th_when_fg then 1
    """

    node_ids_to_classes = {}
    H, W = gt_mask.shape

    for node_id, node in nodes.items():
        x, y = node 

        if x < 0 or x >= H or y < 0 or y >= W:
            node_ids_to_classes[node_id] = 0
            continue

        if gt_mask[x, y] != 0:
            node_ids_to_classes[node_id] = int(gt_mask[x, y])
            continue

        up = max(0, x - r)
        down = min(x + r + 1, H)
        left = max(0, y - r)
        right = min(y + r + 1, W)

        window = gt_mask[up:down, left:right]
        nonzero = window[window != 0]

        if len(nonzero) >= min_votes:
            values, counts = np.unique(nonzero, return_counts=True)
            node_ids_to_classes[node_id] = int(values[np.argmax(counts)])
        else:
            node_ids_to_classes[node_id] = 0

    return node_ids_to_classes     

def make_image_graph(nodes,edges,labels,binary_mask,img_base,prob_mask,frangi_mask):
    """
    inputs : 
        - nodes: {node_id : (x,y)}
        - edges: {edge_id : [node_id1 , node_id2]}
        - labels: {node_id : label}
        - binary_mask : (H,W)
        - img_base : (C,H,W)
        - prob_mask : (H,W)
        - frangi_mask : (H,W) [0-1]
    outputs : 
        - features : (node_counts,4/5) [x,y,radius,degree,side?]
    """


    edge_index = np.zeros((2,2*len(edges)),dtype=np.int64)
    
    features ,y= feature_extraction(
        binary_mask=binary_mask,
        prob_mask = prob_mask,
        img_base=img_base,
        nodes=nodes,
        edges=edges,
        labels = labels,
        frangi_mask = frangi_mask,
        path=None
    )
    # for node_id in nodes:
    #     feature = nodes[node_id]
    #     label = labels[node_id]
    #     features[node_id] = np.array(feature)
    #     y[node_id] = label

    for edge_id , edge in edges.items():
        id_1 = 2*edge_id 
        id_2 = 2*edge_id + 1

        edge_index[0][id_1] = edge[0]
        edge_index[1][id_1] = edge[1]

        edge_index[0][id_2] = edge[1]
        edge_index[1][id_2] = edge[0]
    return features , edge_index , y

@torch.no_grad()
def create_gnn_single(pred_probs,pred_classes,gt_mask,img_base,frangi_mask,
                      r,min_votes,path=None,
                      low_thresh=0.2,high_thresh=0.7,
                      enhance_mask=False,labeling_mode = "strict"):
    """
    inputs : 
        - pred_probs: (H,W)
        - gt_mask: (H,W)
        - enhance_mask: bool
    """
    # pred_probs = F.softmax(pred_logits,dim=1)[0,1].cpu().numpy() # H,W
    if(enhance_mask):
        pred_mask = hysteresis_threshold(pred_probs, low_thresh=low_thresh, high_thresh=high_thresh) # H,W
    else:
        # pred_mask = (pred_probs>=0.5).astype(np.uint8)
        pred_mask = pred_classes.astype(np.uint8)
        # pred_mask = (gt_mask!=0).astype(np.uint8)
   
    skel = skeletonize(pred_mask,method='lee')
    Xs,Ys = np.where(skel==1)

    nodes,nodes_to_id = build_nodes(Xs=Xs,Ys=Ys)
    edges = build_edges(nodes=nodes,nodes_to_id=nodes_to_id,mask_skel=skel)
    if(labeling_mode=="strict"):
        labels = labeling_nodes(nodes=nodes,gt_mask=gt_mask)
    elif(labeling_mode=="binary_kernel"):
        labels = binary_kernel_based_labeling_nodes(
            nodes = nodes,
            gt_mask=gt_mask,
            r=r,
            min_votes = min_votes
        )
    features , edge_index , y = make_image_graph(
        nodes=nodes,
        prob_mask = pred_probs,
        edges=edges,
        labels=labels,
        binary_mask = pred_mask,
        img_base = img_base,
        frangi_mask=frangi_mask
    )
    
    return features , edge_index , y
def colorize_class_map(class_map, colors, background_color=(0, 0, 0)):
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:] = np.array(background_color, dtype=np.uint8)

    valid = (class_map != -1) & (class_map >= 0) & (class_map < len(colors))
    rgb[valid] = colors[class_map[valid]]

    return rgb
def process_and_save_one_gnn_sample(job):
    (
        pred_probs,
        pred_classes,
        gt_mask,
        img_base,
        frangi_mask,
        side,
        name,
        part,
        features_path,
        labels_path,
        edges_path,
        saved_pred_path,
        save_pred_masks,
        low_thresh,
        high_thresh,
        enhance_mask,
        labeling_mode,
        r,
        min_votes,
        name_prefix,
    ) = job

    features, edge_index, y = create_gnn_single(
        pred_probs=pred_probs,
        pred_classes = pred_classes,
        gt_mask=gt_mask,
        img_base=img_base,
        low_thresh=low_thresh,
        high_thresh=high_thresh,
        enhance_mask=enhance_mask,
        path=None,
        frangi_mask=frangi_mask,
        r=r,
        labeling_mode=labeling_mode,
        min_votes=min_votes
    )

    sides = np.full((len(features), 1), side)
    features = np.concatenate([features, sides], axis=1)

    np.save(os.path.join(features_path, f"{name_prefix}{name}.npy"), features)
    np.save(os.path.join(labels_path, f"{name_prefix}{name}.npy"), y)
    np.save(os.path.join(edges_path, f"{name_prefix}{name}.npy"), edge_index)

    if save_pred_masks:
        # binary_mask = (pred_probs >= 0.5).astype(np.uint8) * 255
        binary_mask = pred_classes.astype(np.uint8)*255
        cv2.imwrite(os.path.join(saved_pred_path, f"{name_prefix}{name}.png"), binary_mask)

    return name

def _process_one_quality_sample(
    data_name,
    part,
    ds_path,
    gt_mask_base,
    image_base,
    output_cleaned_masks,
    colors,
    class_count,
    device,
):
    name_stem = Path(data_name).stem

    pred_mask_base = ds_path / f"model_preds/{part}"

    pred_path = pred_mask_base / f"{name_stem}.png"
    gt_path = gt_mask_base / part / f"{name_stem}.npy"
    image_path = image_base / part / f"{name_stem}.png"

    pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    gt_mask = np.load(str(gt_path)).astype(np.uint8)
    gt_mask[gt_mask>=5] -=15
    gt_mask[gt_mask<5] = 0
    gt_mask[gt_mask>=5] -=4 
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    features = np.load(ds_path / part / "features" / f"{name_stem}.npy")
    labels = np.load(ds_path / part / "labels" / f"{name_stem}.npy")

    features = torch.from_numpy(features).to(device)

    pred_skel_mask = make_skel_from_features(
        features=features.cpu().numpy(),
        pred_mask=pred_mask,
        pred_labels=labels,
        colors=None,
        use_color=False
    )

    node_coords = features[..., 0:2].cpu().numpy()

    multi_TP, multi_FP, multi_FN, final_multi_pred_mask = calaculate_gnn_multi_dice(
        pred_skel_mask=pred_skel_mask,
        pred_mask=pred_mask,
        gt_mask=gt_mask.copy(),
        labels=labels,
        node_coords=node_coords,
        class_count=class_count
    )

    binary_TP, binary_FP, binary_FN, final_binary_pred_mask = calaculate_gnn_binary_dice(
        binary_mask=pred_mask,
        binary_gt_mask=(gt_mask != 0).astype(int),
        labels=(labels != 0).astype(int),
        node_coords=node_coords
    )

    image_wise_multi_dice = (
        (2 * multi_TP + 1e-8) /
        (2 * multi_TP + multi_FP + multi_FN + 1e-8)
    )
    valid_multi_classes = (multi_TP + multi_FP + multi_FN) > 0
    valid_multi_classes[0] = False 

    image_wise_binary_dice = (
        (2 * binary_TP + 1e-8) /
        (2 * binary_TP + binary_FP + binary_FN + 1e-8)
    )

    gt_image = img.copy()
    pred_image = img.copy()
    cleaned_image = img.copy()

    gt_image = draw_mask(
        gt_image,
        gt_mask,
        training_mode="multi",
        colors=colors
    )

    pred_image = draw_mask(
        pred_image,
        pred_mask // 255,
        training_mode="multi",
        colors=colors
    )

    cleaned_image = draw_mask(
        cleaned_image,
        final_multi_pred_mask,
        training_mode="multi",
        colors=colors
    )

    full_plot = draw_fast_plots(
        [img, gt_image, pred_image, cleaned_image],
        plot_shape=(2, 2)
    )

    cv2.imwrite(
        str(output_cleaned_masks / part / f"{name_stem}.png"),
        full_plot
    )

    return {
        "name_stem": name_stem,

        "multi_TP": multi_TP.cpu(),
        "multi_FP": multi_FP.cpu(),
        "multi_FN": multi_FN.cpu(),

        "binary_TP": binary_TP.cpu(),
        "binary_FP": binary_FP.cpu(),
        "binary_FN": binary_FN.cpu(),

        "image_wise_multi_dice": image_wise_multi_dice[valid_multi_classes].mean().item(),
        "image_wise_binary_dice": image_wise_binary_dice[1].item(),
    }


@torch.no_grad()
def calaculate_dataset_quality(
    gt_mask_base,
    image_base,
    ds_path,
    class_map= None,
    max_workers=4,
):
    parts = ["train", "val", "test"]

    if(class_map == None):
        class_map = {
            0: "bg", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "6", 7: "7", 8: "8",
            9: "9", 10: "9a", 11: "10", 12: "10a",
            13: "11", 14: "12", 15: "12a", 16: "13",
            17: "14", 18: "14a", 19: "15", 20: "16",
            21: "16a", 22: "16b", 23: "16c",
            24: "12b", 25: "14b"
        }
    class_count = len(class_map)
    results = {
        "class name": [class_map[i] for i in range(len(class_map))] + [
            "binary score",
            "mean score"
        ],
        "dice train": [],
        "recall train": [],
        "precision train": [],
        "dice val": [],
        "recall val": [],
        "precision val": [],
        "dice test": [],
        "recall test": [],
        "precision test": []
    }

    

    colors = np.array([
        (242,  24,  24),
        (242,  77,  24),
        (242, 129,  24),
        (242, 181,  24),
        ( 24, 242, 216),
        (242, 234,  24),
        (146,  24, 242),
        (199, 242,  24),
        (146, 242,  24),
        ( 94, 242,  24),
        (242,  24, 181),
        ( 42, 242,  24),
        ( 94,  24, 242),
        ( 24, 242,  59),
        (242,  24, 129),
        ( 24, 242, 111),
        ( 24, 242, 164),
        ( 24, 164, 242),
        (199,  24, 242),
        ( 24, 216, 242),
        ( 24, 111, 242),
        (242,  24, 234),
        ( 24,  59, 242),
        ( 42,  24, 242),
        (242,  24,  77),
    ], dtype=np.uint8)

    ds_path = Path(ds_path)
    gt_mask_base = Path(gt_mask_base)
    image_base = Path(image_base)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_cleaned_masks = ds_path / "cleaned_masks"
    os.makedirs(output_cleaned_masks, exist_ok=True)

    for part in parts:
        image_wise_dice_dict = {
            "images": [],
            "multi_dice": [],
            "binary_dice": []
        }
        total_multi_TP = torch.zeros(class_count)
        total_multi_FP = torch.zeros(class_count)
        total_multi_FN = torch.zeros(class_count)

        total_binary_TP = torch.zeros(2)
        total_binary_FP = torch.zeros(2)
        total_binary_FN = torch.zeros(2)

        pred_mask_base = ds_path / f"model_preds/{part}"
        data_names = os.listdir(pred_mask_base)

        os.makedirs(output_cleaned_masks / part, exist_ok=True)

        worker_args = [
            (
                data_name,
                part,
                ds_path,
                gt_mask_base,
                image_base,
                output_cleaned_masks,
                colors,
                class_count,
                device,
            )
            for data_name in data_names
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            outputs = list(
                tqdm(
                    executor.map(lambda args: _process_one_quality_sample(*args), worker_args),
                    total=len(worker_args),
                    desc=f"Processing {part}"
                )
            )

        for out in outputs:
            total_multi_TP += out["multi_TP"]
            total_multi_FP += out["multi_FP"]
            total_multi_FN += out["multi_FN"]

            total_binary_TP += out["binary_TP"]
            total_binary_FP += out["binary_FP"]
            total_binary_FN += out["binary_FN"]

            image_wise_dice_dict["images"].append(out["name_stem"])
            image_wise_dice_dict["multi_dice"].append(out["image_wise_multi_dice"])
            image_wise_dice_dict["binary_dice"].append(out["image_wise_binary_dice"])

        multi_dice_score = (
            (2 * total_multi_TP + 1e-8) /
            (2 * total_multi_TP + total_multi_FP + total_multi_FN + 1e-8)
        )

        multi_recall_score = (
            (total_multi_TP + 1e-8) /
            (total_multi_TP + total_multi_FN + 1e-8)
        )

        multi_precision_score = (
            (total_multi_TP + 1e-8) /
            (total_multi_TP + total_multi_FP + 1e-8)
        )

        binary_dice_score = (
            (2 * total_binary_TP + 1e-8) /
            (2 * total_binary_TP + total_binary_FP + total_binary_FN + 1e-8)
        )

        binary_recall_score = (
            (total_binary_TP + 1e-8) /
            (total_binary_TP + total_binary_FN + 1e-8)
        )

        binary_precision_score = (
            (total_binary_TP + 1e-8) /
            (total_binary_TP + total_binary_FP + 1e-8)
        )

        for i in range(len(multi_dice_score)):
            dice = float(multi_dice_score[i])
            recall = float(multi_recall_score[i])
            precision = float(multi_precision_score[i])

            results[f"dice {part}"].append(dice)
            results[f"recall {part}"].append(recall)
            results[f"precision {part}"].append(precision)

        results[f"dice {part}"].append(binary_dice_score[1].numpy().mean())
        results[f"dice {part}"].append(multi_dice_score[1:].numpy().mean())

        results[f"recall {part}"].append(binary_recall_score[1].numpy().mean())
        results[f"recall {part}"].append(multi_recall_score[1:].numpy().mean())

        results[f"precision {part}"].append(binary_precision_score[1].numpy().mean())
        results[f"precision {part}"].append(multi_precision_score[1:].numpy().mean())

        image_wise_dice_df = pd.DataFrame(image_wise_dice_dict)
        image_wise_dice_df.to_csv(ds_path / f"{part}_image_wise_dice.csv", index=False)

    for key , val in results.items():
        print(key,len(val))
    results_df = pd.DataFrame(results)
    results_df.to_csv(ds_path / "dataset_dice.csv", index=False)
    

    display(results_df)

@torch.no_grad()
def create_gnn_dataset_parallel_limited(
    model,
    dataloaders,
    args,
    base_path,
    r,
    min_votes,
    paths,
    class_map = None,
    enhance_mask=False,
    low_thresh=0.2,
    high_thresh=0.7,
    save_pred_masks=True,
    labeling_mode="strict",
    max_workers=4,
    max_pending=8,
    calc_ds_quality = True,
    name_prefix = ""
):
    saved_pred_base = os.path.join(base_path, "model_preds")
    os.makedirs(saved_pred_base, exist_ok=True)
    base_path = Path(base_path)
    model.eval()
    
    base_dataset_Path = Path(args["paths"]["base_path"])

    for part, dataloader in dataloaders.items():
        out_path = os.path.join(base_path, part)
        features_path = os.path.join(out_path, "features")
        labels_path = os.path.join(out_path, "labels")
        edges_path = os.path.join(out_path, "edges")
        saved_pred_path = os.path.join(saved_pred_base, part)
        
        augment_ds_path_base = Path(base_dataset_Path/ args["paths"]["gnn_dataset_masks"] )
        if (augment_ds_path_base / part).exists() and (augment_ds_path_base / part).is_dir():
            shutil.rmtree(augment_ds_path_base)

        os.makedirs(out_path, exist_ok=True)
        os.makedirs(features_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)
        os.makedirs(edges_path, exist_ok=True)
        os.makedirs(saved_pred_path, exist_ok=True)

        os.makedirs(augment_ds_path_base /"images" / part, exist_ok=True)
        os.makedirs(augment_ds_path_base /"labels" / part, exist_ok=True)

        pending = set()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pbar = tqdm(dataloader, desc=f"{part}")

            for data in pbar:
                
                img_base = data["new_image_not_normal"]
                img = data["new_image"]
                mask = data["mask"]
                side = data["side"]
                name = data["name"]

                if(5 not in mask):
                    continue

                mask[mask<5] = 0
                mask[mask>=5]-=4
                img = img.to(args["device"], non_blocking=True)

                name = name[0]
                side = side[0]
                img_base_np = img_base[0].numpy()
                gt_mask = mask.numpy()[0]

                # with torch.autocast(device_type=args["device"], dtype=torch.float16):
                #     pred_logits = model(img)[0]
                #     pred_probs = F.softmax(pred_logits, dim=1)[0,-1].cpu().numpy()
                #     pred_classes = torch.argmax(pred_logits,dim=1)[0].cpu().numpy()

                #     pred_classes[pred_classes!=5] = 0
                #     pred_classes[pred_classes!=0] = 1

                pred_probs = gt_mask.astype(float)
                pred_classes = gt_mask.copy()
                pred_classes[pred_classes!=0] = 1
                pred_probs[0,0] = 1
                pred_classes[0,0] = 1
                if(pred_classes.sum()==0):
                    # print(f"missed 16 branch in img {name}")
                    continue
                frangi_mask = np.load(os.path.join(
                    paths["base_path"],
                    paths["filter_folder"],
                    part,
                    name + ".npy"
                ))

                job = (
                    pred_probs,
                    pred_classes,
                    gt_mask.copy(),
                    img_base_np,
                    frangi_mask,
                    side,
                    name,
                    part,
                    features_path,
                    labels_path,
                    edges_path,
                    saved_pred_path,
                    save_pred_masks,
                    low_thresh,
                    high_thresh,
                    enhance_mask,
                    labeling_mode,
                    r,
                    min_votes,
                    name_prefix
                )

                pending.add(executor.submit(process_and_save_one_gnn_sample, job))

                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
                np.save(augment_ds_path_base / "labels" / part /f"{name_prefix}{name}.npy",gt_mask)
                cv2.imwrite(augment_ds_path_base / "images" / part /f"{name_prefix}{name}.png",img.cpu().numpy()[0].transpose(1,2,0))
            # finish remaining jobs
            for f in tqdm(pending, desc=f"finishing {part}"):
                f.result()
    if(calc_ds_quality):
        calaculate_dataset_quality(
            gt_mask_base = Path(paths["base_path"])/paths["labels_folder"],
            image_base = Path(paths["base_path"])/paths["images_folder"],
            ds_path = base_path,
            class_map = class_map
        )