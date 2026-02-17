import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
import matplotlib.pyplot as plt
from skimage.filters import frangi , hessian , meijering
import random
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
        
# Augementations 
def normalize_xca(image):
    x = image.astype(np.float32, copy=False)
    m = x > 0
    if np.any(m):
        mean = x[m].mean()
        std  = x[m].std()
        x[m] = (x[m] - mean) / (std + 1e-8)
        x[~m] = 0.0
    else:
        x = x / 1.0


    return {"image":x}

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