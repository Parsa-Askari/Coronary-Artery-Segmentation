import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import timm
class HRNET(nn.Module):
    def __init__(self,args):
        super(HRNET,self).__init__()
        self.model = smp.Unet(
            encoder_name="tu-hrnet_w32",
            encoder_weights=None,
            in_channels=args["in_c"],
            classes=args["class_count"], 
        )
    def forward(self,x):
        return [self.model(x)]
if __name__ == "__main__":
    ls = torch.rand((2,3,512,512)).to("cuda")
    model = HRNET(args={"in_c":3,"class_count":2}).to("cuda")
    print(model(ls)[0].shape)