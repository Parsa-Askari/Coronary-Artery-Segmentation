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
    def __init__(self,kernel_size = (50, 50)):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    def __call__(self,img):
        neg_img = cv2.bitwise_not(img)
        tophat_img = cv2.morphologyEx(neg_img, cv2.MORPH_TOPHAT, self.kernel,borderType=cv2.BORDER_REPLICATE)
        return cv2.subtract(img, tophat_img)
class DoubleClahe:
    def __init__(self,kernel_size=(15, 15), clip_limit=2.0, tile_grid=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_size)
        
    def __call__(self,img):
        clah_1 = self.clahe.apply(img)
        img_inv = cv2.bitwise_not(clah_1)
        top_hat = cv2.morphologyEx(img_inv,cv2.MORPH_TOPHAT,self.kernel)
        img_sub = cv2.subtract(img_inv,top_hat)
        clah_2 = self.clahe.apply(img_sub)
        return clah_2
class Meijering:
    def __init__(self,sigmas=[1,2,3]):
        self.sigmas = sigmas
    def __call__(self,img):
        return meijering(img,sigmas=self.sigmas)
    
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
def crop_with_bbox(img, mask, binary_mask ,bbox):
    x1, y1, x2, y2 = map(int, bbox)

    img_c = img[y1:y2, x1:x2, ...]
    mask_c = mask[y1:y2, x1:x2, ...]
    binary_mask_c = binary_mask[y1:y2, x1:x2, ...]
    return img_c, mask_c, binary_mask_c

def morph_binary_mask(x, **kwargs):
    m = x.copy()
    
    if m.shape[-1] == 3:
        
        m2 = cv2.cvtColor(m,cv2.COLOR_RGB2GRAY)
    else:
        m2 = m

    m2 = (m2 > 0).astype(np.uint8)

    if np.random.rand() < 0.5:
        k = np.ones((3, 3), np.uint8)
       
        if np.random.rand() < 0.5:
            m2 = cv2.dilate(m2, k, iterations=1)
        else:
            m2 = cv2.erode(m2, k, iterations=1)

    if np.random.rand() < 0.5:
        blurred = cv2.GaussianBlur(m2.astype(np.float32), (3, 3), 0)
        m2 = (blurred > 0.5).astype(np.uint8)

    if np.random.rand() < 0.5:
        h, w = m2.shape        
        for _ in range(200):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            m2[y, x] = 0

    
    if m.shape[-1] == 3:
        m_out = cv2.cvtColor(m[...,0],cv2.COLOR_GRAY2RGB)
    else:
        m_out = m2

    return m_out

def laplacian_pyramid_enhance(img, levels=3):

    img = img.astype(np.float32) / 255.0

    G = [img]
    for i in range(1, levels):
        G.append(cv2.pyrDown(G[i-1]))

    L = []
    for i in range(levels - 1):
        GE = cv2.pyrUp(G[i+1], dstsize=(G[i].shape[1], G[i].shape[0]))
        L.append(G[i] - GE)

    H = np.array([[0, -1,  0],
                  [-1, 5, -1],
                  [0, -1,  0]], dtype=np.float32)

    L_enh = [cv2.filter2D(Li, -1, H) for Li in L]

    current = G[-1]
    for i in reversed(range(levels - 1)):
        current = cv2.pyrUp(current, dstsize=(L_enh[i].shape[1], L_enh[i].shape[0]))
        current = current + L_enh[i]

    current = np.clip(current, 0, 1)
    return (current * 255).astype(np.uint8)

def gaussian_diffrential_scale_inverse(img,sigma_small=1.0,sigma_large=3.0,k=0.5):
    img_norm = (img).astype(np.uint8)/255.5
    mu = cv2.GaussianBlur(img_norm,(0,0),sigma_small)
    sq = cv2.GaussianBlur(img_norm**2 , (0,0),sigma_small)
    local_std = np.sqrt(np.maximum(sq - mu**2, 0)) + 1e-6

    lc = (img_norm - mu)/local_std

    lc_norm = (lc - lc.min())/(lc.max()-lc.min() + 1e-6)

    g_small = cv2.GaussianBlur(img_norm , (0,0),sigma_small)
    g_big = cv2.GaussianBlur(img_norm , (0,0),sigma_large)

    contrast_weights = np.clip(0.5 + k * (lc_norm - 0.5), 0, 1)
    enhanced = contrast_weights * g_small + (1 - contrast_weights) * g_big

    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced*255).astype(np.uint8)
def _clip_like_input(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # nnU-Net typically works in float32, but this keeps uint8/uint16 safe if you use them.
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

if __name__ == "__main__" :
    # img = cv2.imread("./dataset/syntax/images/train/10.png",cv2.IMREAD_GRAYSCALE)

    # trans1 = DoubleClahe(kernel_size=(3,3),tile_grid=(3,3))
    # trans2 = DoubleClahe(kernel_size=(3,3),tile_grid=(5,5)) 
    # trans3 = DoubleClahe(kernel_size=(3,3),tile_grid=(8,8))
    # trans4 = DoubleClahe(kernel_size=(3,3),tile_grid=(10,10))  

    # plt.figure(figsize=(10,10))
    # plt.subplot(3,3,1)
    # plt.imshow(cv2.cvtColor(trans1(img),cv2.COLOR_GRAY2RGB))
    # plt.subplot(3,3,2)
    # plt.imshow(cv2.cvtColor(trans2(img),cv2.COLOR_GRAY2RGB))
    # plt.subplot(3,3,3)
    # plt.imshow(cv2.cvtColor(trans3(img),cv2.COLOR_GRAY2RGB))
    # plt.subplot(3,3,4)
    # plt.imshow(cv2.cvtColor(trans4(img),cv2.COLOR_GRAY2RGB))
    # plt.subplot(3,3,5)
    # plt.imshow(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
    # plt.show()
    print(morph_binary_mask(np.random.rand(448, 448, 3).astype(np.uint8)).shape)
