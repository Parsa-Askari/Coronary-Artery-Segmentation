from torchvision.models.swin_transformer import (
    swin_t,swin_s,swin_b,
    swin_v2_t,swin_v2_s,swin_v2_b,
    Swin_S_Weights,Swin_V2_S_Weights,
    Swin_B_Weights,Swin_V2_B_Weights,
    Swin_T_Weights,Swin_V2_T_Weights
)
from transformers import AutoModelForSemanticSegmentation , AutoConfig
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
        abs_class_count = args["abs_class_count"]
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
                abs_class_count = abs_class_count,
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

class SwinUperNet3(nn.Module):
    def __init__(self,args):
        super(SwinUperNet3,self).__init__()
        model_name = "openmmlab/upernet-swin-base"
        # config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        # config.backbone_config.window_size = 10

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name,
            local_files_only=True,
            # ignore_mismatched_sizes=True,
            # config=config
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=args["swin"]["encoder_out_channels"],
            out_channels=args["class_count"],
            kernel_size=1
        )
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=7+3,out_channels=3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        class_count = args["class_count"]
        if(args["deep_super_vision"]):
            self.dsv0_conv = nn.Conv2d(in_channels=512,out_channels=class_count,kernel_size=1)
            self.dsv1_conv = nn.Conv2d(in_channels=512,out_channels=class_count,kernel_size=1)
            self.dsv2_conv = nn.Conv2d(in_channels=512,out_channels=class_count,kernel_size=1)
        
        self.deep_super_vision = args["deep_super_vision"]
        self.features = {}
        self._set_hooks()

    def _get_hook(self, name):
        def hook(model, input, output):
            self.features[name] = output
        return hook

    def _set_hooks(self):

        head = self.model.decode_head
        head.lateral_convs[0].register_forward_hook(self._get_hook('lat0'))
        head.lateral_convs[1].register_forward_hook(self._get_hook('lat1'))
        head.lateral_convs[2].register_forward_hook(self._get_hook('lat2'))

    def forward(self,x,gt_mask=None,ep=None):
        outs = []
        z = self.input_conv(x)
        z = self.model(z).logits
        if(self.deep_super_vision):
            out0 = self.dsv0_conv(self.features['lat0'])
            out1 = self.dsv1_conv(self.features['lat1'])
            out2 = self.dsv2_conv(self.features['lat2'])
            outs = [out0,out1,out2]
        z = self.conv1x1(z)
        return [z] + outs , 0

class SwinUperNet2(nn.Module):

    def __init__(self, args):
        super().__init__()
        base_c = 10
        class_count = args["class_count"] 
        encoder_out_c = args["swin"]["encoder_out_channels"]
        self.warmup_ep = args["swin"]["warmup_ep"]
        self.class_count = class_count
        model_name = "openmmlab/upernet-swin-base"

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name,
            local_files_only=True,

        )
        self.compresser = nn.Conv2d(encoder_out_c, base_c, kernel_size=1)

        def head(in_c):
            return nn.Sequential(
                nn.Conv2d(in_c, base_c, kernel_size=3, padding=1),
                nn.GroupNorm(2, base_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_c, 1, kernel_size=1),
            )

        self.classifier_5  = head(base_c) 
        self.classifier_6  = head(base_c + 1)
        self.classifier_7  = head(base_c + 1)
        self.classifier_8  = head(base_c + 1)
        self.classifier_11 = head(base_c + 1)
        self.classifier_13 = head(base_c + 1)
        self.add_bg_c      = nn.Conv2d(base_c, 1, kernel_size=1)


    @staticmethod
    def _cond(pred_logit, gt_ch, p_gt):
        if p_gt <= 0.0:
            return torch.sigmoid(pred_logit)
        if p_gt >= 1.0:
            return gt_ch
        use_gt = torch.rand(pred_logit.shape[0], 1, 1, 1,
                            device=pred_logit.device) < p_gt
        return torch.where(use_gt, gt_ch, torch.sigmoid(pred_logit))

    def forward(self, x, gt_mask=None, ep=None):
        z = self.compresser(self.model(x).logits)

        if self.training and gt_mask is not None and ep is not None:
            p_gt = max(0.0, 1.0 - ep / max(1, self.warmup_ep))
        else:
            p_gt = 0.0

        gt_oh = None
        if gt_mask is not None:
            gt_oh = F.one_hot(gt_mask, self.class_count).permute(0, 3, 1, 2).to(z.dtype)

        out_5  = self.classifier_5(z)
        c5     = self._cond(out_5,  gt_oh[:, 1:2] if gt_oh is not None else None, p_gt)

        out_6  = self.classifier_6(torch.cat([c5, z], 1))
        c6     = self._cond(out_6,  gt_oh[:, 2:3] if gt_oh is not None else None, p_gt)

        out_7  = self.classifier_7(torch.cat([c6, z], 1))
        c7     = self._cond(out_7,  gt_oh[:, 3:4] if gt_oh is not None else None, p_gt)

        out_8  = self.classifier_8(torch.cat([c7, z], 1))

        out_11 = self.classifier_11(torch.cat([c5, z], 1))
        c11    = self._cond(out_11, gt_oh[:, 5:6] if gt_oh is not None else None, p_gt)

        out_13 = self.classifier_13(torch.cat([c11, z], 1))

        out_bg = self.add_bg_c(z)
        channels = [out_bg, out_5, out_6, out_7, out_8, out_11, out_13]
        out = torch.cat(channels, dim=1)

        aux = None
        if gt_oh is not None:
            aux = sum(
                F.binary_cross_entropy_with_logits(logit, gt_oh[:, i:i + 1])
                for i, logit in enumerate(channels)
            )

        outs = []

        return [out] , aux


class SwinUperNet(nn.Module):
    def __init__(self,args):
        super(SwinUperNet,self).__init__()
        model_name = "openmmlab/upernet-swin-base"
        # config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        # config.backbone_config.window_size = 10

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name,
            local_files_only=True,
            # ignore_mismatched_sizes=True,
            # config=config
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=args["swin"]["encoder_out_channels"],
            out_channels=args["class_count"],
            kernel_size=1
        )
        class_count = args["class_count"]
        if(args["deep_super_vision"]):
            self.dsv0_conv = nn.Conv2d(in_channels=512,out_channels=class_count,kernel_size=1)
            self.dsv1_conv = nn.Conv2d(in_channels=512,out_channels=class_count,kernel_size=1)
            self.dsv2_conv = nn.Conv2d(in_channels=512,out_channels=class_count,kernel_size=1)
        
        self.deep_super_vision = args["deep_super_vision"]
        self.features = {}
        self._set_hooks()

    def _get_hook(self, name):
        def hook(model, input, output):
            self.features[name] = output
        return hook

    def _set_hooks(self):

        head = self.model.decode_head
        head.lateral_convs[0].register_forward_hook(self._get_hook('lat0'))
        head.lateral_convs[1].register_forward_hook(self._get_hook('lat1'))
        head.lateral_convs[2].register_forward_hook(self._get_hook('lat2'))

    def forward(self,x,gt_mask=None,ep=None):
        outs = []
        z = self.model(x).logits
        if(self.deep_super_vision):
            out0 = self.dsv0_conv(self.features['lat0'])
            out1 = self.dsv1_conv(self.features['lat1'])
            out2 = self.dsv2_conv(self.features['lat2'])
            outs = [out0,out1,out2]
        z = self.conv1x1(z)
        return [z] + outs , 0
    
if __name__ == "__main__":
    model = SwinUperNet2(
        args={"swin":{"encoder_out_channels":150,"warmup_ep":3},
              "deep_super_vision":True,"class_count":7}).cuda()

    x = torch.zeros(1,3,512,512).cuda()
    outs = model(x,3)
    for o in outs:
        print(o.shape)
