import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
from torchview import draw_graph
img_path = "./dataset/syntax/images/train/9.png"


if __name__ == "__main__":
    inp = torch.zeros(2,3,224,224)
    model = models.resnet34(pretrained=True)
    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4 
    print(layer1)
    print("=========")
    print(layer2)
    print("=========")
    print(layer3)
    print("=========")
    print(layer4)

    # graph = draw_graph(
    #     model,
    #     input_data=inp,
    #     expand_nested=True,
    #     save_graph=True,
    #     filename="resnet34_model_graph",
    #     directory="./"
    # )