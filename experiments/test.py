import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = "./dataset/syntax/images/train/9.png"

img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img,cmap="gray")
plt.show()
