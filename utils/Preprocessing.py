import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

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
def normalize_xca(img, **kwargs):
    x = img.astype(np.float32, copy=False)
    m = x > 0
    if np.any(m):
        mean = x[m].mean()
        std  = x[m].std()
        x[m] = (x[m] - mean) / (std + 1e-8)
        x[~m] = 0.0
    else:
        x = x / 1.0
    return x