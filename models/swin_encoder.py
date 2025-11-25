from torchvision.models.swin_transformer import (
    swin_t,swin_s,swin_b,
    swin_v2_t,swin_v2_s,swin_v2_b,
    Swin_S_Weights,Swin_V2_S_Weights,
    Swin_B_Weights,Swin_V2_B_Weights,
    Swin_T_Weights,Swin_V2_T_Weights
)
from transformers import AutoModelForSemanticSegmentation
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
###IE###
from .swin_blocks import *
###SS###
class SwinEncoder(nn.Module):
    def __init__(self,args):
        self.confs = {
            "swin_t" : [swin_t,224,96,4,Swin_T_Weights],
            "swin_s" : [swin_s,224,96,4,Swin_S_Weights],
            "swin_b" : [swin_b,224,128,4,Swin_B_Weights],
            "swin_v2_t" : [swin_v2_t,256,96,4,Swin_V2_T_Weights],
            "swin_v2_s" : [swin_v2_s,256,96,4,Swin_V2_S_Weights],
            "swin_v2_b" : [swin_v2_b,256,128,4,Swin_V2_B_Weights]
        }
        super(SwinEncoder,self).__init__()
        swin_head = args["swin_head"]
        swin_type = args["swin_type"]
        class_count = args["class_count"]
        in_c = args["in_c"]
        deep_super_vision  = args["deep_super_vision"]
        swin_builder , base_img_size , emb_size , depth , weight_fn = self.confs[swin_type]
        self.backbone = swin_builder(weights=weight_fn.IMAGENET1K_V1)
        # print(self.backbone)
        if(in_c!=3):
            self.convert_base_channels(in_c)
        input_h,input_w =args["image_shape"] 
        if(swin_head=="costume"):
            self.head = CostumeHead(
                input_h=input_h,
                input_w=input_w,
                depth=depth,
                emb_size=emb_size,
                class_count = class_count,
                deep_super_vision=deep_super_vision

            )
    def convert_base_channels(self,in_c):
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1,keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias[:] = old_conv.bias
        self.backbone.features[0][0] = new_conv

    def forward(self,x):
        z = self.backbone.features(x)
        z = self.backbone.norm(z)
        z = self.backbone.permute(z)
        return self.head(z)

class SwinUperNet(nn.Module):
    def __init__(self,args):
        super(SwinUperNet,self).__init__()
        model_name = "openmmlab/upernet-swin-base"
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.conv1x1 = nn.Conv2d(
            in_channels=150,
            out_channels=args["class_count"],
            kernel_size=1
        )
    def forward(self,x):
        z = self.model(x).logits
        z = self.conv1x1(z)
        return [z]
if __name__ == "__main__":
    args = {
        "base_path" : "../arcade/nnUnet_dataset/syntax",
        "in_c" : 3,
        "base_channel" :32,
        "image_shape" : (448,448),
        "class_count" : 26 ,
        "attention" : True,
        "k":40,
        "batch_size" : 10,
        "num_workers" : 10,
        "device" : "cuda" if torch.cuda.is_available() else "cpu",
        "lr" : 0.01,
        "momentum" : 0.99,
        "weight_decay" : 3e-5,
        "epcohs":200,
        "f_int_scale" : 2,
        "full_report_cycle" : 10,
        "max_channels":512,
        "input_channels":1,
        "unet_depth":6,
        "loss_type":"tversky loss",
        "alpha":0.3,
        "beta":0.7,
        "t_gamma":2.0,
        "f_gamma":2.0,
        "f_loss_scale":1,
        "loss_coefs":{"CE":1.0,"Second":1.0},
        "swin_head" : "costume",
        "swin_type":"swin_s",
        "output_base_path" : "./outputs",
        "name" : "Attention7-DSV-tev-new_augs-binary2color-FCE",
        "deep_super_vision" : True,
        "f_alpha":None
    }
    model = SwinEncoder(args)
    inp = torch.rand((1,3,args["image_shape"][0],args["image_shape"][1]))
    outs = model(inp)
    for out in outs:
        print(out.shape)