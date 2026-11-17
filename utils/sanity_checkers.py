import matplotlib.pyplot as plt
import numpy as np
###IE###
from .helpers import denormalize
###SS###
def check_seq_dataloader(train_loader):
    for imgs , masks , stop_labels , full_masks in train_loader:
        img = imgs[0].permute(1,2,0).numpy()
        img =denormalize(img)

        full_mask = full_masks[0].numpy()

        mask = masks[0].numpy()

        plt.figure(figsize=(20,20))
        plt.subplot(4,4,1)
        plt.imshow(img)
        plt.subplot(4,4,2)
        plt.imshow(full_mask)
        n_class = mask.shape[0]
        for i in range(n_class):
            print(np.unique(mask[i]))
            plt.subplot(4,4,i+3)
            plt.imshow(mask[i])
        print(masks.shape)
        masks = masks.reshape(-1,*masks.shape[2:])
        mask = masks[0:masks.shape[0]//2]
        print(mask.shape)
        plt.figure(figsize=(20,20))
        plt.subplot(4,4,1)
        plt.imshow(img)
        plt.subplot(4,4,2)
        plt.imshow(full_mask)
        n_class = mask.shape[0]
        for i in range(n_class):
            print(np.unique(mask[i]))
            plt.subplot(4,4,i+3)
            plt.imshow(mask[i])
        
        break