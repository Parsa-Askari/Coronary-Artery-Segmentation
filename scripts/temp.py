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

from gnn_helpers import (
    fix_16_family , find_right_root , 
    set_labels , find_dist_from_root , 
    sample_at_nodes , find_root,
    find_degree , find_radius,
    fit_feature_stats,load_feature_stats,
    apply_feature_stats,find_branch_depth,
    left_mapping_function,find_left_root
)
from gnn_coloring import compute_gnn_ceiling

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
        masks,
        base_output_path,
        augment_ds_path_base,
        part,
        root_idx_dict,
        index_maps,
        root_location,
        branch_index = -1,
        t=None,
        image_shape=512,
        k=10,
        side="both",
        device = "cuda",
        prefix = ""
    ):
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

    for mask,pred_mask,name_stem in tqdm(masks):
        name = f"{name_stem}.npy"
        if(t is not None):
            r = t(image = np.ones((512, 512)),mask=mask)
            mask = r["mask"]

        idx_to_keep = index_maps[branch_index]
        gt_mask = np.zeros_like(mask)
        for i,idx in enumerate(idx_to_keep) :
            gt_mask[mask==idx] = i+1
        

        binary_mask = (pred_mask==branch_index).astype(int).copy()
        binary_mask[-1,-1] = 1

        coords, ei , pred_skel = extract_graph(pred_mask=binary_mask,gt_mask=gt_mask,k=k)
        root_idx = find_left_root(coords,gt_mask,branch_idx=1,
                                  root_idx=root_idx_dict,axis=root_location)

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

if __name__ == "__main__":
    old_class_map = {
        1: '5',2: '6',3: '7',4: '8',
        5: '9',6: '9a',7: '10',8: '10a',
        9: '11',10: '12',11: '12a',12: '13',
        13: '14',14: '14a',15: '15',
        16: '12b',17: '14b'
    }
    validator_class_maps = {
        1:{0:"bg",1:"5",2:"12"},
        2:{0:"bg",1:"6",2:"9"}, #6,9
        3:{0:"bg",1:"7",2:"9a",3:"10"},#7,9a,10
        4:{0:"bg",1:"8",2:"10a"}, #8,10a
        5:{0:"bg",1:"11",2:"12a",3:"12b"},#11,12a,12b
        6:{0:"bg",1:"13",2:"14",3:"14a",4:"14a",5:"15"},#13,14,14a,14b,15
    }
    class_map = {
        1: '5', 2: '6', 3: '7', 4: '8', 5: '11', 6: '13'
    }
    root_locations = {
        1 : "left",
        2 : "left",
        3 : "topleft",
        4 : "topleft",
        5 : "topleft",
        6 : "topleft",
    }

    

    index_maps = {
        1:[1,10], #5,12
        2:[2,5], #6,9
        3:[3,6,7],#7,9a,10
        4:[4,8], #8,10a
        5:[9,11,16],#11,12a,12b
        6:[12,13,14,17,15],#13,14,14a,14b,15
    }
    reversed_map = {v: k for k, v in class_map.items()}
    root_idx_dict = {
        1:[1], # 5 : 5 itself is the root
        2:[1,2],#6 : connection point of 5 and 6 or highest point of 6
        3:[2,3],#7 : connection point of 6 and 7 or highest point of 7
        4:[3,4],#6 : connection point of 7 and 8 or highest point of 8
        5:[1,5],#6 : connection point of 11 and 5 or highest point of 11
        6:[5,6] #6 : connection point of 13 and 11 or highest point of 13
    }
    lut = left_mapping_function(
        old_class_map = old_class_map,
        new_class_map = class_map
    )
    class_count = len(class_map) + 1
    #1,2,3,4,5,
    classes = [6]
    for c in tqdm(classes):

        branch = class_map[c]
        branch_index = reversed_map.get(branch,-1)

        transforms = {
            "raw_":A.Compose([],additional_targets={'v_mask': 'mask',"expert_mask":"mask","topology_mask":"mask"}),
        }
        
        right = [1,2,3,4,20,21,22,23]

        base_input_path       = Path("./dataset/syntax/labels")
        base_output_path      = Path(f"./dataset/syntax/gnn_dataset/k_gt_dataset_branch[{branch}]")
        base_output_aug_masks = Path(f"./dataset/syntax/gnn_dataset_augmented_images_branch[{branch}]")

        masks = {"train":[],"val":[],"test":[]}
        missed = {}
        for part in ["train","val","test"]:
            names = os.listdir(base_input_path / part)
            missed[part] = []
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
                if(branch_index not in pred_mask):
                    continue
                if(branch_index in lut[mask] and branch_index not in pred_mask):
                    missed[part].append(name)

                masks[part].append([mask,pred_mask,name_stem])
                

        
        if (Path(base_output_path)).exists() and (Path(base_output_path)).is_dir():
                shutil.rmtree(base_output_path)
        if base_output_aug_masks.exists() and base_output_aug_masks.is_dir():
                shutil.rmtree(base_output_aug_masks)

        os.makedirs(base_output_aug_masks,exist_ok=True)
        for prefix,t in transforms.items():
            build_gnn_gt_dataset(
                masks = masks["train"],
                base_output_path=base_output_path,
                augment_ds_path_base=base_output_aug_masks,
                side="right", k=1,
                prefix = prefix,
                t = t,
                part="train",
                branch_index = branch_index,
                root_idx_dict=root_idx_dict,
                index_maps = index_maps,
                root_location=root_locations[c]
            )
        for part in ["val","test"]:
            build_gnn_gt_dataset(
                masks = masks[part],
                base_output_path=base_output_path,
                augment_ds_path_base=base_output_aug_masks,
                side="right", k=1,
                prefix = "",
                t = None,
                part=part,
                branch_index = branch_index,
                root_idx_dict=root_idx_dict,
                index_maps = index_maps,
                root_location=root_locations[c]
            )
    
        compute_gnn_ceiling(

            gt_root = Path(base_output_aug_masks) / "labels",
            pred_root = Path(base_output_path) / "model_preds",
            gnn_root = base_output_path,
            # class_map = {0:"bg",1:"1",2:"2",3:"3",4:"4",5:"16",6:"16a",7:"16b",8:"16c"}
            class_map = validator_class_maps[c]
        )
        out = Path(base_output_path)  
        stats_path = out / "stats.npz"
        with open(out / "dataset_keys.json","w") as f:
            json.dump(dataset_keys,f)


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














    import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import torch
from tqdm.notebook import tqdm
import datetime
import os
import json
import shutil
import cv2
import zarr
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import confusion_matrix , classification_report
import torch.nn.functional as F
from pathlib import Path
from IPython.display import display
###IE###
from utils.helpers import (
    compute_gnn_confution_matrix,reconstruct_multi_vessels,
    TP_TN_FP_FN,reconstruct_binary_vessels,make_skel_from_features,
    calaculate_gnn_multi_dice ,calaculate_gnn_binary_dice,
    ensemble_pred )
from scripts.build_notebook import build_kaggle_project
from utils.graph_extraction import colorize_class_map
###SS###
@torch.no_grad()
def save_full_report(recorder,dataloaders,model,
                     output_base_path,training_mode,
                     side,family_branch,
                     name,args,class_map,chosen_index,class_names):
    
    if(training_mode=="binary"):
        colors = np.array([
            (255,0,0),
            (0,255,0),
        ])
    else:
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
            (0  , 255,   0),   # Green
        ], dtype=np.uint8)
    if(not args["ensemble"]):
        model.eval()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder_name = str(now)
    if(name):
        save_folder_name += f" [{name}]"

    output_folder_path = os.path.join(output_base_path,save_folder_name)
    os.makedirs(output_folder_path,exist_ok=True)
    
    all_results = generate_model_results(
        model = model,
        training_mode = training_mode,
        dataloaders = dataloaders,
        args = args
    )

    print("Save Model")
    if(args["ensemble"]):
        for i,m in enumerate(model):
            torch.save(m.state_dict(), os.path.join(output_folder_path,f"model_{i}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(output_folder_path,"model.pth"))
    
    print("Saving Memory")
    save_memory(recorder,args,output_folder_path)

    print("generate report")
    evaluation_report(all_results,output_folder_path,class_names=class_names)
    summary_report(recorder,output_folder_path)
    print("Saving All Plots")
    draw_loss_plots(recorder , output_folder_path)
    draw_avg_metric_plots(recorder , output_folder_path)
    compute_gnn_confution_matrix(
        all_results = all_results,
        class_maps = class_map,
        output_folder_path=output_folder_path
    )
    
    if(training_mode!="side"):
        draw_results(
            model=model,
            output_folder_path=output_folder_path,
            paths=args["paths"],
            class_map=class_map,
            chosen_index=chosen_index,
            colors = colors,
            training_mode = training_mode,
            use_binary_model = args["gnn_simple"]["use_binary_model"],
            args = args,
            side=side,
            family_branch = family_branch
        )
        calculate_final_dice(
            model = model,
            dataloaders=dataloaders,
            paths=args["paths"],
            class_count=args["class_count"],
            output_folder_path = output_folder_path,
            training_mode = args["training_mode"],
            class_map = class_map,
            use_binary_model = args["gnn_simple"]["use_binary_model"],
            args = args
        )
    notebook_out_path = os.path.join(output_folder_path,"notebook.ipynb") 
    shutil.copyfile(f"./notebooks/Gnn_Main.ipynb",notebook_out_path )
    print("builfding kaggle project")
    build_kaggle_project(output_folder_path,notebook_name="Gnn_Main.ipynb")

def calculate_final_dice(model,dataloaders,paths,class_count,output_folder_path,
                         training_mode,class_map,args,use_binary_model=False):
    if(not args["ensemble"]):
        model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt_mask_base = paths["gt_mask_folder"]
    out_p = Path(output_folder_path)

    base_path = paths["base_path"]

    for part , dataloader in dataloaders.items():
        total_TP =  torch.zeros(class_count)
        total_FP = torch.zeros(class_count)
        total_FN = torch.zeros(class_count)

        pred_mask_base = os.path.join(
            base_path,
            f"model_preds/{part}",
        )
        part_results = {"class":[],"dice":[]}
        for data , labels , name  in tqdm(dataloader):
            pred_path = os.path.join(pred_mask_base,name[0]+".png")
            gt_path = os.path.join(gt_mask_base,part,name[0]+".npy")

            pred_mask = cv2.imread(pred_path,cv2.IMREAD_GRAYSCALE)

            gt_mask = (np.load(gt_path)).astype(np.uint8)
            # gt_mask[gt_mask>=5]-=15
            # gt_mask[gt_mask<5]=0
            # gt_mask[gt_mask>=5]-=4

            features = data.x.squeeze(0).to(device)
            edge_index = data.edge_index.squeeze(0).to(device)
            batch = data.batch.squeeze(0).to(device)
            labels = data.y.squeeze(0).numpy()
            if(args["ensemble"]):
                pred_labels = ensemble_pred(
                    models=model,
                    model_inputs=[features,edge_index,batch],
                    device = device
                )
                pred_labels = torch.argmax(pred_labels,dim=1)
                pred_labels = pred_labels.reshape(-1).cpu().numpy()
            else:
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    pred_labels = model(features,edge_index,batch)
                    pred_labels = torch.argmax(pred_labels,dim=1)
                    pred_labels = pred_labels.reshape(-1).cpu().numpy()
            
            pred_skel_mask = make_skel_from_features(
                features = features.cpu().numpy(),
                pred_mask = pred_mask,
                pred_labels = pred_labels,
                colors = None,
                use_color = False
            )
            
            node_coords = features[...,0:2].cpu().numpy()
            if(training_mode=="binary"):
                TP , FP , FN ,_= calaculate_gnn_binary_dice(
                    binary_mask = pred_mask,
                    binary_gt_mask = (gt_mask!=0).astype(int),
                    labels = (pred_labels!=0).astype(int),
                    node_coords = node_coords
                )
            else:
                TP , FP , FN  ,_= calaculate_gnn_multi_dice(
                    pred_skel_mask = pred_skel_mask,
                    pred_mask = pred_mask,
                    gt_mask = gt_mask,
                    labels = pred_labels,
                    node_coords = node_coords,
                    class_count = class_count
                )
            total_TP += TP
            total_FN += FN
            total_FP += FP
            
            
        dice_score = (2 * total_TP + 1e-8) / (2 * total_TP + total_FN + total_FP + 1e-8)
        print(f"dice {part}:",dice_score,dice_score[1:].mean())
        for key , class_name in class_map.items():
            part_results["class"].append(class_name)
            part_results["dice"].append(float(dice_score[key]))
        part_results["class"].append("mean dice")
        part_results["dice"].append(float(dice_score[1:].numpy().mean()))
        part_r_df = pd.DataFrame(part_results)
        part_r_df.to_csv(out_p / f"dice_results_{part}.csv",index=False)
        display(part_r_df)
        print("--------")
        
def summary_report(recorder,output_folder_path):
    report_path = os.path.join(output_folder_path,"summary_report.txt")
    full_metric_dict = recorder.full_metric_dict
    
    full_report = []
    for part in full_metric_dict:
        best_index = np.argmax(full_metric_dict[part]["f1"])
        part_report = f"Best <{part}> Epoch : {best_index+1}\n"
        for key , avg_metrics in full_metric_dict[part].items():
            avg_m = avg_metrics[best_index]
            part_report += f"{key} : {avg_m}\n"
        full_report.append(part_report)
    full_report = ("+=+"*50).join(full_report)

    with open(report_path,"w") as f:
        f.write(full_report)

@torch.no_grad()
def generate_model_results(model,training_mode,dataloaders,args):
    if(not args["ensemble"]):
        model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for part , dataloader in dataloaders.items():
        all_labels = []
        all_preds = []
        results[part]={}
        for data , labels , name  in dataloader:
            if(training_mode=="side"):
                labels = labels.to(device)
                imgs = data.to(device)
                features = None
                edge_index = None
            else:
                features = data.x.squeeze(0).to(device)
                edge_index = data.edge_index.squeeze(0).to(device)
                labels = data.y.squeeze(0).to(device)
                batch = data.batch.squeeze(0).to(device)
                imgs = None
            if(args["ensemble"]):
                pred_logits = ensemble_pred(
                    models=model,
                    model_inputs=[features,edge_index,batch],
                    device = device
                )
            else:
                with torch.autocast(device_type=device,dtype=torch.float16):
                    if(training_mode=="side"):
                        pred_logits = model(imgs)
                    else:
                        pred_logits = model(features,edge_index,batch)

            preds = torch.argmax(pred_logits,dim=1)
            
            preds = preds.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            
            all_labels += labels
            all_preds += preds

        results[part]["pred"] = all_preds
        results[part]["true"] = all_labels
    return results
def evaluation_report(all_results,output_folder_path,class_names):
    for part ,results in all_results.items():
        part_path = os.path.join(output_folder_path,f"{part}_report.json")
        all_labels = results["true"]
        all_preds = results["pred"]
        report = classification_report(
            y_true=all_labels,
            y_pred=all_preds,
            zero_division=0,
            labels=list(range(0,len(class_names))),
            target_names=class_names
        )
        dict_report = classification_report(
            y_true=all_labels,
            y_pred=all_preds,
            output_dict=True,
            zero_division=0,
            labels=list(range(0,len(class_names))),
            target_names=class_names
        )
        print(f"{part} evaluation Report")
        print("=+="*30)
        print(report)

        with open(part_path , "w") as f:
            json.dump(dict_report,f,indent=4)
        
def save_memory(recorder,args,output_folder_path):
    loss_history_path = os.path.join(output_folder_path,"loss_history.json")
    avg_metric_path = os.path.join(output_folder_path,"avg_metric_hostory.json")
    args_path = os.path.join(output_folder_path,"args.json")

    with open(loss_history_path , "w") as f:
        json.dump(recorder.full_loss_dict,f,indent=4)
    with open(avg_metric_path , "w") as f:
        json.dump(recorder.full_metric_dict,f,indent=4)
    with open(args_path , "w") as f:
        json.dump(args,f,indent=4)

def draw_loss_plots(recorder , output_folder_path):
    plt_path =os.path.join(output_folder_path,"loss_plot.png")

    loss_history = recorder.full_loss_dict
    loss_count = len(recorder.losses_keys)

    colors = plt.cm.tab20(np.linspace(0, 1, loss_count))
    plt.figure(figsize=(15,20))
    for i,part in enumerate(loss_history):
        plt.subplot(2,1,i+1)
        for j,(loss_name,data) in enumerate(loss_history[part].items()):
            length = len(data)-1
            x = np.arange(length)
            plt.plot(x,data[:-1],color = colors[j],label=loss_name)
        plt.title(f"{part} loss plot")
        plt.legend() 
    plt.savefig(plt_path,dpi=150)

def draw_avg_metric_plots(recorder , output_folder_path):
    plt_path =os.path.join(output_folder_path,"metric_plot.png")

    metric_history = recorder.full_metric_dict
    metric_count = len(recorder.metric_keys)

    colors = plt.cm.tab20(np.linspace(0, 1, metric_count))
    plt.figure(figsize=(15,20))
    for i,part in enumerate(metric_history):
        plt.subplot(2,1,i+1)
        for j,(metric_name,data) in enumerate(metric_history[part].items()):
            length = len(data)-1
            x = np.arange(length)
            plt.plot(x,data[:-1],color = colors[j],label=metric_name)
        plt.title(f"{part} loss plot")
        plt.legend() 
    plt.savefig(plt_path,dpi=150)

@torch.no_grad()
def draw_results(model,output_folder_path,paths,class_map,side,family_branch,
                 chosen_index,colors,training_mode,use_binary_model,args):
    if(side == 0):
        image_names = {
            "hard":[
                "3","8","14","15","23","24","34",
                "47","55","58","78","118","124",
                "126"
            ]
        }
    elif(side == 1):
        family_images = {
            "5":[
                "1","2","5","13","40","43","51",
                "63","93","94","104","107","122",
                "137","152","153"
            ]
        }
        image_names = {
            "hard":family_images[family_branch]
        }
    if(training_mode=="multi" and use_binary_model):
        base_path = paths["multi_base_path"]
    else:
        base_path = paths["base_path"]

    features_base_path = os.path.join(
        base_path,
        "val",
        paths["features_folder"],
    )
    features_base_path = os.path.join(
        base_path,
        "val",
        paths["features_folder"],
    )
    index_base_path = os.path.join(
        base_path,
        "val",
        paths["edges_folder"],
    )
    labels_base_path = os.path.join(
        base_path,
        "val",
        paths["labels_folder"],
    )

    preds_base_path = os.path.join(
        base_path,
        "model_preds/val",
    )
    gt_base_path = os.path.join(
        paths["gt_mask_folder"],
        "val",
    )
    binary_base = os.path.join(
        paths["binary_result_folder"] , 
        "val"
    )
    if(not args["ensemble"]):
        model.eval()

    for diff , images in image_names.items():
        plt_path = os.path.join(output_folder_path,f"{diff}_examples.png")
        plt.figure(figsize=(25,70))
        patches = [
            mpatches.Patch(color=np.array(colors[j]) / 255.0, label=class_map[j])
            for j in range(0,len(class_map))
        ]
        h = len(images)
        img_index = 1
        for i,image in enumerate(tqdm(images)):
            if(image+".npy" not in os.listdir(features_base_path)):
                continue
            feature_path = os.path.join(features_base_path,image+".npy")
            edge_index_path = os.path.join(index_base_path,image+".npy")
            labels_path = os.path.join(labels_base_path,image+".npy")
            pred_path = os.path.join(preds_base_path,image+".png")
            gt_path = os.path.join(gt_base_path,image+".npy")

            features = np.load(feature_path)[...,chosen_index]
            edge_index = np.load(edge_index_path)
            labels = np.load(labels_path).reshape(-1).astype(int)

            
            
            features_t = torch.from_numpy(features).float().to("cuda")
            edge_index_t = torch.from_numpy(edge_index).int().to("cuda")

            if(args["ensemble"]):
                pred_labels = ensemble_pred(
                    models=model,
                    model_inputs=[features_t,edge_index_t],
                    device="cuda"
                )
                pred_labels = torch.argmax(pred_labels,dim=1)
                pred_labels = pred_labels.reshape(-1).cpu().numpy()
            else:
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    pred_labels = model(features_t,edge_index_t)
                    pred_labels = torch.argmax(pred_labels,dim=1)
                    pred_labels = pred_labels.reshape(-1).cpu().numpy()
            


            pred_mask = cv2.imread(pred_path,cv2.IMREAD_GRAYSCALE)
            gt_mask = (np.load(gt_path)).astype(np.uint8)
            # gt_mask[gt_mask>=5]-=15
            # gt_mask[gt_mask<5]=0
            # gt_mask[gt_mask>=5]-=4

            gt_skel_mask = make_skel_from_features(features,pred_mask,
                                                     labels,colors,use_color=True).astype(np.uint8)
            
            pred_skel_mask = make_skel_from_features(features,pred_mask,
                                                     pred_labels,colors,use_color=True).astype(np.uint8)
            
            node_coords = features[...,0:2]
            if(training_mode=="binary"):
                new_pred_mask = (reconstruct_binary_vessels(
                        binary_mask = pred_mask//255,
                        node_coords = node_coords,
                        node_labels = pred_labels
                )*255).astype(np.uint8)
            elif(training_mode=="multi"):
                full_class_map = reconstruct_multi_vessels(
                    vessel_mask = pred_mask//255,
                    skeleton_colored = pred_skel_mask,
                    node_coords = node_coords,
                    node_labels = pred_labels,
                    remove_color = True
                )
                # print(np.unique(full_class_map))
                new_pred_mask = colorize_class_map(
                    class_map = full_class_map,
                    colors = colors
                )
                # new_pred_mask[full_class_map==0]=[0,0,0]
            plt.subplot(h,5,img_index)
            if(training_mode!="multi"):
                plt.imshow(gt_mask)
                plt.title(f"gt mask{image}")
            else:
                new_gt_mask = colors[gt_mask]
                new_gt_mask[gt_mask==0]=(0,0,0)

                u_classes = map(str,np.unique(gt_mask).tolist())
                s_classes = "|".join(u_classes)

                plt.imshow(new_gt_mask)
                plt.title(f"gt mask{image}:{s_classes}")

            plt.subplot(h,5,img_index+1)
            plt.imshow(pred_mask)
            plt.title(f"pred mask{image}")
            
            plt.subplot(h,5,img_index+2)
            plt.imshow(gt_skel_mask)
            plt.title(f"gt skel {image}")
            
            plt.subplot(h,5,img_index+3)
            plt.imshow(pred_skel_mask)
            plt.title(f"pred skel {image}")
            
            plt.subplot(h,5,img_index+4)
            plt.imshow(new_pred_mask)
            plt.title(f"new pred mask {image}")

            img_index+=5
            if(img_index-1==5):
                plt.legend(
                    handles=patches,
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    borderaxespad=0.,
                    title="Classes"
                )
        plt.savefig(plt_path)
