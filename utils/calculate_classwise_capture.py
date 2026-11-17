import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import os

def classwise_capture(best_model,data_loaders,model_name,args,out_folder="model_analysis"):
    multi_class_map = {
        1: '1',2: '2', 3: '3',4: '4',
        5: '5',6: '6',7: '7',8: '8',
        9: '9',10: '9a',11: '10',12: '10a',
        13: '11',14: '12',15: '12a',16: '13',
        17: '14',18: '14a',19: '15',20: '16',
        21: '16a',22: '16b',23: '16c',
        24: '12b',25: '14b'
    }

    base_ds_path = Path(args["paths"]["base_path"])
    save_root = base_ds_path / out_folder / model_name
    os.makedirs(save_root, exist_ok=True)

    labels_base = Path(args["paths"]["base_path"])  / args["paths"]["labels_folder"] 

    df = {"part":[]}
    for i in range(1,26):
        df[multi_class_map[i]]=[]

    with torch.no_grad():
        for part , dl in data_loaders.items():
            labels_part = labels_base / part
            total = np.zeros(25)
            captured =np.zeros(25)

            for data in tqdm(dl):
                img =data["new_image"].cuda()
                name = data["name"][0]
                mask = np.load(labels_part / f"{name}.npy")
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    pred_mask,_ = best_model(img)
                    pred_mask = pred_mask[0]
                    pred_mask = torch.argmax(pred_mask,dim=1)
                pred_mask = pred_mask.cpu().numpy()[0]
                un = np.unique(mask)
                un = un[un!=0]
                for u in un:
                    gt_binary_mask = np.zeros_like(mask)
                    gt_binary_mask[mask==u] = 1

                    intersection = np.sum(np.logical_and(gt_binary_mask,pred_mask))
                    all = np.sum(mask==u)
                    total[u-1] += all
                    captured[u-1] += intersection
            
            captured = captured / total
            for i in range(len(captured)):
                df[multi_class_map[i+1]].append(captured[i])
            df["part"].append(part)
        df =pd.DataFrame(df)
        df = df.T
        df.to_csv(save_root / "class_wise_ratio.csv")