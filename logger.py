import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import torch
from tqdm.notebook import tqdm
import datetime
import os
import json
import shutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
###IE###
from utils.helpers import (
    draw_mask , compute_confution_matrix ,no_teacher_forcing_pipeline,
    process_masks , make_example_datasets , denormalize
)
from build_notebook import build_kaggle_project
###SS###
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


@torch.no_grad()
def save_full_report(recorder,output_base_path,model,valid_loader,
                     args,class_map,valid_images,test_transforms,
                     class_count,device,name=None,notebook_name="Multi_Main"):
    now = datetime.datetime.now()
    save_folder_name = str(now)
    if(name):
        save_folder_name += f" [{name}]"
    output_folder_path = os.path.join(output_base_path,save_folder_name)

    os.makedirs(output_folder_path,exist_ok=True)
    
    print("Save Model")
    torch.save(model.state_dict(), os.path.join(output_folder_path,"model.pth"))

    print("Saving Memory")
    save_memory(recorder,args,output_folder_path)

    print("Saving All Plots")
    draw_loss_plots(recorder , output_folder_path)
    draw_avg_metric_plots(recorder , output_folder_path)
    draw_all_metric_plots(recorder , output_folder_path)
    compute_confution_matrix(
        data_loader=valid_loader,
        model = model,
        class_maps = class_map,
        draw_plot = True,
        class_count=len(class_map)+1,
        output_folder_path=output_folder_path
    )
    print("Saving Examples")
    draw_examples(
        model = model,
        args = args,
        class_map = class_map,
        valid_images = valid_images,
        test_transforms = test_transforms,
        output_folder_path = output_folder_path,
        class_count=class_count,
        device=device
    )

    print("Saving Verbal Results")
    write_verbal_results(recorder,output_folder_path)

    print("Copying Notebook To Results")

    notebook_out_path = os.path.join(output_folder_path,"notebook.ipynb") 
    shutil.copyfile(f"./{notebook_name}",notebook_out_path )
    print("builfding kaggle project")
    build_kaggle_project(output_folder_path,notebook_name=notebook_name)

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
            
        report+="\n"
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
def draw_examples(model,args,class_map,valid_images,
                  test_transforms,output_folder_path,
                  class_count,device,w=6,h=5):
    image_names = {
        "easy":[
            "3","7","12","9","46","66","101","1",
            "10","35","43","49","70","80","84"
        ],
        "normal":[
            "15","23","47","78","2","17","18","26","30",
            "39","45","48","60","99","102"
        ],
        "hard":[
            "8","14","24","58","4","16","20","21","32", 
            "36","63","69","93","107","154"
        ],
        "very hard":[
            "79","132","136","148","161","22","119","44",
            "129","162","163","182","187","153","95"
        ]
    }
    dataloader_set = make_example_datasets(
        valid_images=valid_images,
        image_names=image_names,
        transform=test_transforms
    )
    model.eval()
    for diff , dalaloader in dataloader_set.items():
        plt_path = os.path.join(output_folder_path,f"{diff}_examples.png")
        plt.figure(figsize=(30,30))
        patches = [
            mpatches.Patch(color=np.array(colors[j-1]) / 255.0, label=class_map[j])
            for j in range(1,len(class_map)+1)
        ]
        img_index = 1
        for imgs , masks , stop_labels , full_masks in tqdm(dalaloader):
            seq_len = masks.shape[1]
            b_size = imgs.shape[0]
            masks = masks.reshape(-1,*masks.shape[2:])

            imgs = imgs.to(device)
            masks = masks.to(device)
            stop_labels = stop_labels.to(device)
            with torch.autocast(device_type=args["device"],dtype=torch.float16):
                pred_stop_labels,pred_masks =  model(imgs,seq_len)
            masks = masks.cpu()
            full_masks = full_masks.to(device)
            pred_masks  = process_masks(
                pred_targets=pred_masks,
                b_size=b_size,
                t_max=seq_len,
                target_shape=(masks.shape[1],masks.shape[2]),
                class_count=class_count
            )
            full_masks = full_masks.cpu().numpy()
            
            pred_masks = pred_masks.cpu().numpy()
            pred_masks = np.argmax(pred_masks,axis=1) # B x H x W

            imgs = imgs.permute(0,2,3,1) # B x H x W x C
            img = imgs[0].cpu().numpy() # H x W x C
            img = denormalize(img)

            full_mask = full_masks[0] # H x W
            pred_mask = pred_masks[0] #H x W
            

            real_annoted = draw_mask(img,full_mask,args,colors)
            pred_annoted = draw_mask(img,pred_mask,args,colors)
            
            plt.subplot(h,w,img_index)
            plt.imshow(real_annoted)
            plt.title(f"Ground Truth ")
            plt.subplot(h,w,img_index+1)
            plt.imshow(pred_annoted)
            plt.title(f"Predicted ")
            img_index+=2
            if(img_index-1==w):
                plt.legend(
                    handles=patches,
                    bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    borderaxespad=0.,
                    title="Classes"
                )
        plt.savefig(plt_path)
        