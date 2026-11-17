import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
import matplotlib.pyplot as plt
from skimage.filters import frangi , hessian , meijering
import random
from skimage import morphology
import albumentations as A
import torch
from torch_geometric.utils import dropout_edge
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from albumentations.core.transforms_interface import DualTransform
from tqdm.auto import tqdm
import pandas as pd
import os
###IE###
###SS###


class CLAHE : 
    def __init__(self,clipLimit=2.0,tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    def __call__(self,img):
        enhanced = np.clip(img, 0, 255)
        return self.clahe.apply(enhanced)
class WhiteTopHat:
    def __init__(self,kernel_size = (50, 50),turn_neg = True):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.turn_neg = turn_neg
    def __call__(self,img):
        neg_img = cv2.bitwise_not(img)
        tophat_img = cv2.morphologyEx(neg_img, cv2.MORPH_TOPHAT, self.kernel,borderType=cv2.BORDER_REPLICATE)
        # tophat_img = morphology.white_tophat(neg_img, self.kernel) 
        return cv2.subtract(img, tophat_img)

class EasyMaskGuided(DualTransform):
    def __init__(self, guide_key="easy_mask", masks_to_edit=("v_mask",), p=1.0):
        super().__init__(p=p)
        self.guide_key = guide_key
        self.masks_to_edit = tuple(masks_to_edit)     # was set(...) -> not JSON serializable

    def get_transform_init_args_names(self):
        return ("guide_key", "masks_to_edit", )

    @property
    def targets_as_params(self):
        return [self.guide_key]  

    def get_params_dependent_on_data(self, params, data):
        return {"region": data[self.guide_key][...,0]}

    def apply_with_params(self, params, **kwargs):
        region = params["region"]
        res = {}
        for key, arg in kwargs.items():
            if arg is None:
                res[key] = arg
            elif key in ("image", "images"):
                res[key] = self._edit_image(arg, region)
            elif key in self.masks_to_edit:
                res[key] = self._edit_mask(arg, region)
            else:
                res[key] = arg    
        return res

    def _edit_image(self, img, region):
        out = img.copy()
        out[region==1] = 255.0
        return out
    def _edit_mask(self, mask, region):
        out = mask.copy()
        out[region==1] = 0
        return out 

def improve_contrast(source):
    img = source[:,:,0]
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))  
    img_not = cv2.bitwise_not(img)  
    se = np.ones((50,50), np.uint8)  
    wth = morphology.white_tophat(img_not, se)  
    raw_minus_topwhite = img.astype(int) - wth 
    raw_minus_topwhite = ((raw_minus_topwhite>0)*raw_minus_topwhite).astype(np.uint8)  
    img = clahe.apply(raw_minus_topwhite) 
    img = img[:,:,np.newaxis] 
    d3 = np.concatenate([img, img, img], axis=2).astype(np.uint8)

    return d3

 

 
 
def improve_contrast_new(img):
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    SE = np.ones((50, 50), np.uint8)
    img = CLAHE.apply(img)                              # 1. CLAHE
    inv = cv2.bitwise_not(img)                          # 2. invert
    wth = morphology.white_tophat(inv, SE)              # 3. white top-hat, large SE
    out = img.astype(int) - wth                         # 4. subtract from the original
    out = ((out > 0) * out).astype(np.uint8)            # 5. non-negative threshold
    out = CLAHE.apply(out)                              # 6. CLAHE again
    return np.stack([out] * 3, axis=2)                  # 7. three channels


# Augementations 
def normalize_xca(image):
    x = image.astype(np.float32, copy=False)
    for c in range(x.shape[-1]):
        ch = x[..., c]
        m = ch > 0

        if np.any(m):
            mean = ch[m].mean()
            std = ch[m].std()
            ch[m] = (ch[m] - mean) / (std + 1e-8)
            ch[~m] = 0.0
        else:
            ch[:] = ch[:]/1.0

    return {"image":x}

def normalize_empty(image):
    return {"image":image}

def _clip_like_input(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if np.issubdtype(ref.dtype, np.integer):
        info = np.iinfo(ref.dtype)
        out = np.clip(out, info.min, info.max)
        return out.astype(ref.dtype, copy=False)
    return out.astype(ref.dtype, copy=False)

class BrightnessMultiplicativeNNUNet2D(A.ImageOnlyTransform):
    def __init__(self, multiplier_range=(0.70, 1.3), p=0.15, always_apply=False):
        super().__init__(always_apply=always_apply, p=p)
        self.multiplier_range = multiplier_range

    def apply(self, img, **params):
        factor = random.uniform(*self.multiplier_range)
        out = img.astype(np.float32) * factor  # multiply intensities
        return _clip_like_input(out, img)

    def get_transform_init_args_names(self):
        return ("multiplier_range",)

class ContrastAugmentationNNUNet2D(A.ImageOnlyTransform):
    def __init__(self, contrast_range=(0.65, 1.5), p=0.15, always_apply=False):
        super().__init__(always_apply=always_apply, p=p)
        self.contrast_range = contrast_range

    def apply(self, img, **params):
        factor = random.uniform(*self.contrast_range)
        x = img.astype(np.float32)

        if x.ndim == 2:  # grayscale HxW
            mean = x.mean()
            out = (x - mean) * factor + mean
        else:            # HxWxC (also fine for grayscale HxWx1)
            mean = x.mean(axis=(0, 1), keepdims=True)  # per-channel mean
            out = (x - mean) * factor + mean

        return _clip_like_input(out, img)

    def get_transform_init_args_names(self):
        return ("contrast_range",)
    
def get_preprocessing(training_mode,args):
    if(training_mode=="cleaning"):
        train_transforms = A.Compose([
            ]  
        )
        test_transforms = A.Compose([
            ]  
        )
    else:
        IGNORE_VAL = 0   # sentinel: "unknown", not "background"

        train_transforms = A.Compose([
            A.Affine(
                translate_percent=(-0.20, 0.20),
                scale=(0.8, 1.25),              # was (0.5, 1.5) -- caliber is discriminative
                rotate=(-30, 30),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=IGNORE_VAL,           # was 0 -> vacated area is UNKNOWN, not background
                p=0.5,
            ),

            A.OneOf([
                A.ElasticTransform(alpha=300.0, sigma=8.0, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=(-0.5, 0.5), p=1.0),
                A.OpticalDistortion(distort_limit=(-0.4, 0.4), p=1.0),
            ], p=0.6),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.25, p=1.0),
                A.RandomGamma(gamma_limit=(70, 140), p=1.0),
            ], p=0.5),

            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0),
                            per_channel=False, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 1.0), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.5),

            # CoarseDropout REMOVED -- it blacked out vessels and labelled them background,
            # the exact contradictory supervision this whole project is fixing.
            # EasyMaskGuided already does a principled version of occlusion.

            A.HorizontalFlip(p=0.5),            # regularisation; worth ablating

        ], additional_targets={'v_mask': 'mask', "expert_mask": "mask",
                            "easy_mask": "mask", "topology_mask": "mask"})

        test_transforms = A.Compose([
            # A.Resize(448,448)
            ],
            additional_targets={'v_mask': 'mask',"expert_mask":"mask","easy_mask":"mask","topology_mask":"mask"}
        )
    return train_transforms , test_transforms

def gnn_cord_jitter(cords,cord_smoother=0.005):
    noise = torch.rand_like(cords)*cord_smoother
    cords = cords + noise
    cords = torch.clamp(cords,0.0,1.0)
    return cords

def noisy_feature_jitter(features,prob_idx,radius_idx,frangi_idx,
                         prob_smoother,radius_smoother,frangi_smoother):
    if(random.random()<0.5 and prob_idx!=-1):
        features[...,prob_idx] += torch.rand_like(features[...,prob_idx])*prob_smoother
        features[...,prob_idx] = torch.clamp(
            features[...,prob_idx],
            0.0,
            1.0
        )
    if(random.random()<0.5 and radius_idx!=-1):
        features[...,radius_idx] += torch.rand_like(features[...,radius_idx])*radius_smoother
        features[...,radius_idx] = torch.clamp(
            features[...,radius_idx],
            0.0,
            1.0
        )
    if(random.random()<0.5 and frangi_idx!=-1):
        features[...,frangi_idx] += torch.rand_like(features[...,frangi_idx])*frangi_smoother
        features[...,frangi_idx] = torch.clamp(
            features[...,frangi_idx],
            0.0,
            1.0
        )
    return features

def gnn_gaussian_noise(features,g_smoother=0.01):
    noise = torch.rand_like(features)*g_smoother
    features = features + noise
    features = torch.clamp(features,0.0,1.0)
    return features

def gnn_masking(features,mask_prob=0.05):
    mask = torch.rand(features.shape) < mask_prob
    features = torch.where(
        mask,
        torch.zeros_like(features),
        features
    )
    return features

def gnn_edge_drop(edge_idx,e_prob=0.3):
    edge_idx , _= dropout_edge(
        edge_idx,p=e_prob,
        force_undirected=True
    )
    return edge_idx


def to_uint8(img):
    img = np.asarray(img)

    if img.dtype == np.uint8:
        return img


    if img.max() <= 1.0:
        img = img * 255

    return np.clip(img, 0, 255).astype(np.uint8)


def ensure_3_channel(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

    return img


def add_white_border(img, border_size=20):
    return cv2.copyMakeBorder(
        img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )


def make_side_by_side_cv2(img, new_img, border_size=20, gap_size=20):
    img = to_uint8(img)
    new_img = to_uint8(new_img)

    img = ensure_3_channel(img)
    new_img = ensure_3_channel(new_img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)

    if img.shape[0] != new_img.shape[0]:
        scale = img.shape[0] / new_img.shape[0]
        new_w = int(new_img.shape[1] * scale)
        new_img = cv2.resize(new_img, (new_w, img.shape[0]))

    img = add_white_border(img, border_size=border_size)
    new_img = add_white_border(new_img, border_size=border_size)


    separator = np.ones(
        (img.shape[0], gap_size, 3),
        dtype=np.uint8
    ) * 255

    combined = np.hstack([img, separator, new_img])

    combined = add_white_border(combined, border_size=border_size)

    return combined


def remove_boarder(img , black_th = 20,min_vote = 0.01 ,pad=0,image_name=None):
    if(img.ndim == 3):
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    mask = gray > black_th
    vote_row = mask.mean(axis = 1)
    vote_col = mask.mean(axis = 0)

    valid_rows = np.where(vote_row>min_vote)[0]
    valid_cols = np.where(vote_col>min_vote)[0]

    H,W = gray.shape
    if(len(valid_rows) == 0 or len(valid_cols)==0):
        # print(f"fallback on image : {image_name}")
        return img
    
    y1 , y2 = valid_rows[0] , valid_rows[-1]
    x1 , x2 = valid_cols[0] , valid_cols[-1]

    y1 = max(0, y1 - pad)
    y2 = min(H - 1, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(W - 1, x2 + pad)

    gray = gray[y1:y2+1,x1:x2+1]

    return gray

import cv2
import numpy as np

def remove_border_scan(img, black_th=40, max_dark_fraction=0.85, pad=0,image_name=None):

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    H, W = gray.shape

    dark = gray <= black_th

    row_dark_fraction = dark.mean(axis=1)
    col_dark_fraction = dark.mean(axis=0)

    y1 = 0
    while y1 < H and row_dark_fraction[y1] > max_dark_fraction:
        y1 += 1

    y2 = H - 1
    while y2 > y1 and row_dark_fraction[y2] > max_dark_fraction:
        y2 -= 1

    x1 = 0
    while x1 < W and col_dark_fraction[x1] > max_dark_fraction:
        x1 += 1

    x2 = W - 1
    while x2 > x1 and col_dark_fraction[x2] > max_dark_fraction:
        x2 -= 1

    y1 = max(0, y1 - pad)
    y2 = min(H - 1, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(W - 1, x2 + pad)

    return img[y1:y2 + 1, x1:x2 + 1]

def remove_cxa_border(
    img,
    black_th=80,
    dilate_iter=3,
    pad=0,
    add_white_border=0
):
    """
    Remove noisy black CXA border.

    black_th:
        pixels below this are treated as border candidates

    dilate_iter:
        connects broken/noisy dark border regions.
        Increase if border is not fully removed.

    pad:
        keep some pixels around the crop

    add_white_border:
        add white padding after cropping
    """

    original = img.copy()

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    H, W = gray.shape

    # dark-ish pixels, not only exact black
    dark = (gray < black_th).astype(np.uint8)

    # connect broken parts of the dark frame
    kernel = np.ones((5, 5), np.uint8)
    dark = cv2.dilate(dark, kernel, iterations=dilate_iter)

    # flood fill from image edges to find dark border connected to outside
    flood = np.zeros((H + 2, W + 2), np.uint8)
    border_connected = dark.copy()

    # flood-fill every edge pixel that belongs to dark area
    for x in range(W):
        if border_connected[0, x]:
            cv2.floodFill(border_connected, flood, (x, 0), 2)
        if border_connected[H - 1, x]:
            cv2.floodFill(border_connected, flood, (x, H - 1), 2)

    for y in range(H):
        if border_connected[y, 0]:
            cv2.floodFill(border_connected, flood, (0, y), 2)
        if border_connected[y, W - 1]:
            cv2.floodFill(border_connected, flood, (W - 1, y), 2)

    # pixels marked 2 are outside-connected black border
    border_mask = border_connected == 2

    # useful image area = not outside-connected border
    content_mask = ~border_mask

    ys, xs = np.where(content_mask)

    if len(xs) == 0 or len(ys) == 0:
        return original,[0,0,511,511]

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - pad)
    x2 = min(W - 1, x2 + pad)
    y1 = max(0, y1 - pad)
    y2 = min(H - 1, y2 + pad)

    cropped = original[y1:y2 + 1, x1:x2 + 1]

    if add_white_border > 0:
        cropped = cv2.copyMakeBorder(
            cropped,
            add_white_border,
            add_white_border,
            add_white_border,
            add_white_border,
            borderType=cv2.BORDER_CONSTANT,
            value=255 if cropped.ndim == 2 else (255, 255, 255)
        )

    return cropped,[x1,y1,x2+1,y2+1]

def _process_one_border_image(args):
    item, base_folder_path, mode,black_th,min_vote = args
    img, label, v_mask, name_stem = item

    # new_img = remove_border_scan(
    #     img,
    #     black_th=black_th,
    #     max_dark_fraction=min_vote,
    #     image_name=name_stem
    # )
    new_img,cords = remove_cxa_border(
        img,
        black_th=40,
        dilate_iter=2,
        pad=0,
        add_white_border=5
    )
    image_path = os.path.join(base_folder_path, f"{name_stem}.png")

    if mode == "plot":
        combined = make_side_by_side_cv2(
            img,
            new_img,
            border_size=25,
            gap_size=25
        )
        cv2.imwrite(image_path, combined)

    else:
        cv2.imwrite(image_path, new_img)

    return [name_stem,cords]




def remove_black_border_multithread(
    images,
    paths,
    part,
    black_th,
    min_vote,
    mode="plot",
    max_workers=8
):
    images = list(images)

    base_folder_path = os.path.join(
        paths["base_path"],
        paths["border_cords_folder"],
        part
    )

    os.makedirs(base_folder_path, exist_ok=True)

    tasks = [
        (item, base_folder_path, mode,black_th,min_vote)
        for item in images
    ]

    cords_dict = {"name":[],"x1":[],"y1":[],"x2":[],"y2":[],}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(_process_one_border_image, tasks),
                total=len(tasks),
                desc=f"Removing borders [{part}]"
            )
        )
    
    for name , cords in results:
        cords_dict["name"].append(name)
        cords_dict["x1"].append(cords[0])
        cords_dict["y1"].append(cords[1])
        cords_dict["x2"].append(cords[2])
        cords_dict["y2"].append(cords[3])
    
    df = pd.DataFrame(cords_dict)
    df.to_csv(
        os.path.join(base_folder_path,"cords.csv"),
        index=False
    )


