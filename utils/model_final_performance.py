import torch
import torch.nn as nn
import datetime
from tqdm.auto import tqdm
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
###IE###
from .losses import TverskyLoss
###SS###
@torch.no_grad()
def model_performance_per_image(model,dataloader,args,out_folder,
                                valid_path="./data/valid_problem_df.csv"):
    model.eval()
    name = args["name"]
    time_now = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    now = f"[{time_now}]"

    out_folder = os.path.join(out_folder,now+name)
    heat_map_folder = os.path.join(out_folder,"heat_maps")
    pred_folder = os.path.join(out_folder,"probs")

    os.makedirs(out_folder,exist_ok=True)
    os.makedirs(heat_map_folder,exist_ok=True)
    os.makedirs(pred_folder,exist_ok=True)

    dice_loss_fn = TverskyLoss(
        args={"alpha":0.5,"beta":0.5,"gamma":1,"weights":None}
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    softmax = nn.Softmax(dim=1)

    loss_history = {"Name":[],"total loss":[],"Dice":[],"CE":[]}
    for img_base , img , mask , name in tqdm(dataloader):
        
        img = img.to(args["device"])
        mask = mask.to(args["device"])
        name = name[0]
        with torch.autocast(args["device"],dtype=torch.float16):
            pred_masks = model(img)[0]
            pred_probs = softmax(pred_masks)
            
            mask_ce_loss = ce_loss_fn(pred_masks,mask)

            fg_pred_probs = pred_probs[:,1:]
            
            dice_family_loss = dice_loss_fn(
                pred_probs = fg_pred_probs,
                gt = mask.unsqueeze(1).float()
            )
        
            mask_ce_loss = mask_ce_loss.cpu().item()
            dice_family_loss = dice_family_loss.cpu().item()
            total_loss = mask_ce_loss + dice_family_loss

            loss_history["Name"].append(int(name))
            loss_history["CE"].append(mask_ce_loss)
            loss_history["Dice"].append(dice_family_loss)
            loss_history["total loss"].append(total_loss)

        fig = plt.figure(figsize=(30, 10))

        img_base = img_base.detach().cpu().numpy()[0, 0]
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img_base,cmap="gray")
        ax1.axis("off")

        img_base[mask[0].cpu().numpy()==1] = 255

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(img_base,cmap="gray")
        ax2.axis("off")
        
        ax3 = fig.add_subplot(1, 3, 3)

        prob_mask = fg_pred_probs.detach().cpu().numpy()[0, 0]
        im = ax3.imshow(
            prob_mask,
            cmap="hot",
            interpolation="bilinear",
        )
        ax3.axis("off")
        fig.colorbar(im, ax=ax3)

        fig.savefig(os.path.join(heat_map_folder, f"{name}.png"), 
                    bbox_inches="tight", dpi=200)
        plt.close(fig)

        np.save(os.path.join(pred_folder,f"{name}.npy"), prob_mask)
        
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(
        os.path.join(out_folder,"loss_df.csv"),index=False
    )

    valid_df = pd.read_csv(valid_path)
    easiest_total_df = loss_df.sort_values(by="total loss").head(20)
    hardest_total_df = loss_df.sort_values(by="total loss").tail(20)

    easiest_dice_df = loss_df.sort_values(by="Dice").head(20)
    hardest_dice_df = loss_df.sort_values(by="Dice").tail(20)

    easiest_ce_df = loss_df.sort_values(by="CE").head(20)
    hardest_ce_df = loss_df.sort_values(by="CE").tail(20)

    pd.merge(easiest_total_df[["Name","total loss"]],valid_df,
             left_on="Name",right_on="name").to_csv(f"{out_folder}/total_easy.csv",index=False)
    pd.merge(hardest_total_df[["Name","total loss"]],valid_df,
             left_on="Name",right_on="name").to_csv(f"{out_folder}/total_hard.csv",index=False)

    pd.merge(easiest_dice_df[["Name","Dice"]],valid_df,
             left_on="Name",right_on="name").to_csv(f"{out_folder}/dice_easy.csv",index=False)
    pd.merge(hardest_dice_df[["Name","Dice"]],valid_df,
             left_on="Name",right_on="name").to_csv(f"{out_folder}/dice_hard.csv",index=False)

    pd.merge(easiest_ce_df[["Name","CE"]],valid_df,
             left_on="Name",right_on="name").to_csv(f"{out_folder}/ce_easy.csv",index=False)
    pd.merge(hardest_ce_df[["Name","CE"]],valid_df,
             left_on="Name",right_on="name").to_csv(f"{out_folder}/ce_hard.csv",index=False)
    return loss_df

"""
valid_df = pd.read_csv("./data/valid_problem_df.csv")
valid_df

loss_df = pd.read_csv("./data/model_performance/[2026-02-25_20-52-06]binary-tests-base+sobel+swin/loss_df.csv")
easiest_total_df = loss_df.sort_values(by="total loss").head(20)
hardest_total_df = loss_df.sort_values(by="total loss").tail(20)

easiest_dice_df = loss_df.sort_values(by="Dice").head(20)
hardest_dice_df = loss_df.sort_values(by="Dice").tail(20)

easiest_ce_df = loss_df.sort_values(by="CE").head(20)
hardest_ce_df = loss_df.sort_values(by="CE").tail(20)

pd.merge(easiest_total_df[["Name","total loss"]],valid_df,left_on="Name",right_on="name").to_csv("./total_easy.csv",index=False)
pd.merge(hardest_total_df[["Name","total loss"]],valid_df,left_on="Name",right_on="name").to_csv("./total_hard.csv",index=False)

pd.merge(easiest_dice_df[["Name","Dice"]],valid_df,left_on="Name",right_on="name").to_csv("./dice_easy.csv",index=False)
pd.merge(hardest_dice_df[["Name","Dice"]],valid_df,left_on="Name",right_on="name").to_csv("./dice_hard.csv",index=False)

pd.merge(easiest_ce_df[["Name","CE"]],valid_df,left_on="Name",right_on="name").to_csv("./ce_easy.csv",index=False)
pd.merge(hardest_ce_df[["Name","CE"]],valid_df,left_on="Name",right_on="name").to_csv("./ce_hard.csv",index=False)
"""