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
from utils.gnn_evaluation import evaluate_merged
###SS###
@torch.no_grad()
def save_full_report(
    restult_sets,
    training_results,
    args,
    chosen_index,
    class_maps,
    models_label_predictions
    ):
    output_base_path = args["paths"]["output_base_path"]
    
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder_name = str(now)
    name = args["name"]
    save_folder_name += f" [{name}]"
    

    output_folder_path = Path(os.path.join(output_base_path,save_folder_name))
    print(output_folder_path)
    os.makedirs(output_folder_path,exist_ok=True)

    
    for c in training_results:
        print(f"==================>processing [{c}]<==================")
        print("Save Model")
        model = training_results[c]["model"]
        recorder = training_results[c]["recorder"]
        torch.save(model.state_dict(), os.path.join(output_folder_path,f"model_{c}.pth"))
    
        print("Saving Memory")
        save_memory(recorder,args,output_folder_path,family=c)

        print("generate report")
        evaluation_report(
            models_label_predictions,
            output_folder_path,
            class_maps=class_maps,
            family = c
        )
        summary_report(recorder,output_folder_path,family=c)
        print("Saving All Plots")
        draw_loss_plots(recorder , output_folder_path,family=c)
        draw_avg_metric_plots(recorder , output_folder_path,family=c)
        compute_gnn_confution_matrix(
            models_label_predictions = models_label_predictions,
            class_maps = class_maps,
            output_folder_path=output_folder_path,
            family=c
        )

    
    for part,sets in restult_sets.items():
        df = {}
        df["dice"] = []
        df["index"] = []
        print(class_maps)
        overall, per_class = evaluate_merged(
            gnn_preds = sets["gnn_preds"],
            cnn_vessels = sets["cnn_vessels"] ,
            gt_vessels = sets["gt_vessels"] ,
            family_labels = class_maps
        )
        per_class["overall"] = overall

        for key , dice in per_class.items():
            df["dice"].append(dice)
            df["index"].append(key)
        print(df)
        print(overall,per_class)
        df = pd.DataFrame(df)
        df.to_csv(output_folder_path / f"{part}_dice.csv",index=False)
        print(f"dice for {part}")
        display(df)
        # draw_results(
        #     model=model,
        #     output_folder_path=output_folder_path,
        #     paths=args["paths"],
        #     class_map=class_map,
        #     chosen_index=chosen_index,
        #     colors = colors,
        #     training_mode = training_mode,
        #     use_binary_model = args["gnn_simple"]["use_binary_model"],
        #     args = args,
        #     side=side,
        #     family_branch = family_branch
        
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
        
def summary_report(recorder,output_folder_path,family):
    report_path = os.path.join(output_folder_path,f"summary_report_{family}.txt")
    full_metric_dict = recorder.full_metric_dict
    
    full_report = []
    for part in full_metric_dict:
        if(len(full_metric_dict[part]["f1"]) == 0):
            continue
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

def evaluation_report(models_label_predictions,output_folder_path,class_maps,family):

    class_map = class_maps[family]
    class_names = []
    print(class_map)
    for i in range(0,len(class_map)):
        class_names.append(class_map[i])

    for part ,results in models_label_predictions.items():
        
        part_path = os.path.join(output_folder_path,f"{part}_report_{family}.json")
        all_labels = results[family]["true"]
        all_preds = results[family]["pred"]
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
        
def save_memory(recorder,args,output_folder_path,family):
    loss_history_path = os.path.join(output_folder_path,f"loss_history_{family}.json")
    avg_metric_path = os.path.join(output_folder_path,f"avg_metric_hostory_{family}.json")
    args_path = os.path.join(output_folder_path,"args.json")

    with open(loss_history_path , "w") as f:
        json.dump(recorder.full_loss_dict,f,indent=4)
    with open(avg_metric_path , "w") as f:
        json.dump(recorder.full_metric_dict,f,indent=4)
    with open(args_path , "w") as f:
        json.dump(args,f,indent=4)

def draw_loss_plots(recorder , output_folder_path,family):
    plt_path =os.path.join(output_folder_path,f"loss_plot_{family}.png")

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
        plt.title(f"{part} loss plot {family}")
        plt.legend() 
    plt.savefig(plt_path,dpi=150)

def draw_avg_metric_plots(recorder , output_folder_path,family):
    plt_path =os.path.join(output_folder_path,f"metric_plot_{family}.png")

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
        plt.title(f"{part} metric plot {family}")
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
