import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def run():
    pass

if __name__ == "__main__":
    frangi_img = np.load("./dataset/syntax/filters/train/5.npy")
    plt.subplot(1,2,1)
    plt.imshow(frangi_img,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(frangi_img>=0.3,cmap="gray")
    plt.show()