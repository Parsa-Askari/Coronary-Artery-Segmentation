import cv2
import numpy as np
import zarr
import os
from collections import Counter
from tqdm import tqdm
import json
def count_pixels (dataset_path="../arcade/nnUnet_dataset/syntax",class_count=26):
    base_path = os.path.join(dataset_path,"labels/train")
    mask_list = os.listdir(base_path)
    counts = {i:0 for i in range(class_count)}
    for path in tqdm(mask_list):
        mask_path = os.path.join(base_path,path)
        mask = zarr.load(mask_path)
        img_class_counts = Counter(mask.reshape(-1).tolist())

        for key , co in img_class_counts.items():
            counts[key]+=co
    return counts
# counts = count_pixels()
# with open("./data/train_class_counts.json","w") as f:
#     json.dump(counts,f,indent=4)


