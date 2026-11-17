from pathlib import Path
import os
import shutil
from tqdm.notebook import tqdm
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import zarr
import nibabel as nib
from skimage.filters import frangi , hessian , meijering
import SimpleITK as sitk
import pandas as pd
from utils.preprocessing import improve_contrast

def polygons_to_mask(segmentation, width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for poly in segmentation:
        pts = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
        draw.polygon(pts, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)

def make_binary_image (labels,segments,revereversed_map,class_map,W,H):
    binary_image = np.zeros((H,W))
    for i,label in enumerate(labels):
        mask = polygons_to_mask([segments[i]],W,H) # makes the mask only for the class = label
        new_label = revereversed_map[class_map[label+1]] # label + 1 since we set 0 to bg
        binary_image[mask==1] = new_label
    binary_image = binary_image.astype(np.uint8)
    return binary_image

def _process_one(
        item,
        out_img_folder_path, 
        out_seg_folder_path, 
        out_prep_folder_path,
        out_f_filter_folder_path,
        reversed_map,
        class_map,
        cords_df,
        transforms=None
    ):

    img_path, data = item
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    cords = cords_df[cords_df["name"]==int(img_name)]
    x1 = int(cords["x1"].item())
    y1 = int(cords["y1"].item())
    x2 = int(cords["x2"].item())
    y2 = int(cords["y2"].item())

    bboxes = data["bbox_yolo"]
    labels = data["labels"]
    seg = data["segmentation"]
    W = data["W"]
    H = data["H"]

    #image_base_path -> #./dataset/syntax/images/train
    shutil.copyfile(img_path, os.path.join(out_img_folder_path, f"{img_name}.png"))

    binary_img = make_binary_image(labels, seg,reversed_map,class_map, W, H)
    seg_path = os.path.join(out_seg_folder_path, f"{img_name}.zarr")

    zarr.save(seg_path, binary_img)
    if(transforms is not None):
        
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        out_binary_path = os.path.join(out_prep_folder_path,f"{img_name}.png")
        for t in transforms:
            img = t(img)
        cv2.imwrite(out_binary_path,img)
        img = img[...,0]

    vesselness = frangi(img, sigmas=range(4,9,1), 
                        beta=0.3,gamma=None, black_ridges=True)
    vesselness[0:y1,:] = 0
    vesselness[y2:,:] = 0
    vesselness[:,0:x1] = 0
    vesselness[:,x2:] = 0
    
    np.save(out_f_filter_folder_path/f"{img_name}.npy",vesselness)
    return img_name

def create_dataset(task, parts=("train", "val", "test"),
                   parallel=True, executor="process", max_workers=None,transforms=None):
    
    out_image_folder_path = os.path.join(root_out_folder, task, "images") #./dataset/syntax/images 
    out_label_folder_path = os.path.join(root_out_folder, task, "labels") #./dataset/syntax/labels 
    out_preprocessed_folder_path = os.path.join(root_out_folder, task, "preprocessed")#./dataset/syntax/preprocessed 
    out_filter_folder_path = os.path.join(root_out_folder, task, "filters")#./dataset/syntax/filters 
    out_segement_folder_path = out_label_folder_path  #./dataset/syntax/labels 

    task_folder_path = os.path.join(root_raw_folder, task) #./syntax
    annot_folder_path = os.path.join(root_raw_folder, f"{task}-{raw_annot_postfix}")#./syntax-processed
    border_cords_path = Path("./dataset/syntax/boarder_removed")
    if os.path.exists(out_label_folder_path):
        shutil.rmtree(out_label_folder_path)


    Path(out_image_folder_path).mkdir(parents=True, exist_ok=True)
    Path(out_label_folder_path).mkdir(parents=True, exist_ok=True)
    Path(out_preprocessed_folder_path).mkdir(parents=True,exist_ok=True)
    Path(out_filter_folder_path).mkdir(parents=True,exist_ok=True)
    for part in parts:
        print(f"[{task}] {part}")

        in_image_folder_path = os.path.join(task_folder_path, part, "images") #./syntax/train/images
        in_annot_folder_path = os.path.join(annot_folder_path, f"{part}.json") #./syntax-processed/train/images
        in_border_cords_path = border_cords_path / part / "cords.csv"

        out_annot_folder_path = os.path.join(out_label_folder_path, part) #./dataset/syntax/labels/train
        out_seg_folder_path = os.path.join(out_segement_folder_path, part)#./dataset/syntax/labels/train
        out_img_folder_path = os.path.join(out_image_folder_path, part)#./dataset/syntax/images/train
        out_prep_folder_path = os.path.join(out_preprocessed_folder_path, part)#./dataset/syntax/preprocessed/train
        out_f_filter_folder_path = os.path.join(out_filter_folder_path, part)#./dataset/syntax/filters/train

        Path(out_annot_folder_path).mkdir(parents=True, exist_ok=True)
        Path(out_seg_folder_path).mkdir(parents=True, exist_ok=True)
        Path(out_img_folder_path).mkdir(parents=True, exist_ok=True)
        Path(out_prep_folder_path).mkdir(parents=True, exist_ok=True)

        cords_df = pd.read_csv(in_border_cords_path)

        with open(in_annot_folder_path, "r") as f:
            annot_file = json.load(f)

        items = list(annot_file.items())
        
        if not parallel:
            
            for item in tqdm(items):
                _process_one(
                    item,
                    out_img_folder_path=out_img_folder_path,
                    out_seg_folder_path=out_seg_folder_path,
                    out_prep_folder_path = out_prep_folder_path,
                    out_f_filter_folder_path = out_f_filter_folder_path,
                    transforms=transforms,
                    reversed_map = reversed_map,
                    class_map = class_map,
                    cords_df = cords_df
                )
        else:
            
            Executor = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
            with Executor(max_workers=max_workers) as pool:
                
                futures = [
                    pool.submit(
                        _process_one,
                            item,
                            out_img_folder_path=out_img_folder_path,
                            out_seg_folder_path=out_seg_folder_path,
                            out_prep_folder_path = out_prep_folder_path,
                            out_f_filter_folder_path = out_f_filter_folder_path,
                            transforms=transforms,
                            reversed_map = reversed_map,
                            class_map = class_map,
                            cords_df = cords_df
                    )
                    for item in items
                ]
                for f in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        f.result()
                    except Exception as e:
                        print("Worker failed:", repr(e))
                    
    l = syntax_mapings if task == "syntax" else stenosis_mappings
    with open(os.path.join(root_out_folder, f"{task}-classes"), "wb") as fp:
        pickle.dump(l, fp)

if __name__ == "__main__":

    root_raw_folder = "."
    root_out_folder = "./dataset" 

    raw_annot_postfix = "processed"

    syntax_mapings = [
        '1', '2', '3', '4', '5', '6', '7',
        '8', '9', '9a', '10', '10a',
        '11', '12', '12a', '13', '14', '14a', 
        '15', '16', '16a', '16b', '16c', '12b', '14b'
    ]

    class_map = {
        1: '1',2: '2', 3: '3',4: '4',
        5: '5',6: '6',7: '7',8: '8',
        9: '9',10: '9a',11: '10',12: '10a',
        13: '11',14: '12',15: '12a',16: '13',
        17: '14',18: '14a',19: '15',20: '16',
        21: '16a',22: '16b',23: '16c',
        24: '12b',25: '14b'
    }

    reversed_map = {
        '1': 1,  '2': 2,   '3': 3,   '4': 4,
        '5': 5,  '6': 6,   '7': 7,   '8': 8,
        '9': 9,  '9a': 10, '10': 11, '10a': 12,
        '11': 13,'12': 14, '12a': 15,'13': 16,
        '14': 17,'14a': 18,'15': 19,'16': 20,
        '16a': 21,'16b': 22,'16c': 23,
        '12b': 24,'14b': 25
    }
    stenosis_mappings = [
        'stenosis'
    ]

    transforms = [
        improve_contrast
    ]
    create_dataset(
        task = "syntax",
        parallel=True, 
        executor="thread",
        transforms=transforms,
        max_workers=4
    )