import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import torch.nn.functional as F

class UnetPlPl(nn.Module):
    def __init__(self, args):
        class_count = args["class_count"]
        in_c = args["in_c"]
        super(UnetPlPl, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_c,
            classes=class_count,
            activation=None
        )
        # --- deep supervision ---
        blocks = self.model.decoder.blocks
        self.ds_nodes = ["x_0_1", "x_0_2", "x_0_3"]      # intermediate top-row nodes
        self._feats = {}
        for n in self.ds_nodes:
            blocks[n].register_forward_hook(
                lambda m, i, o, name=n: self._feats.__setitem__(name, o)
            )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.model(torch.zeros(1, in_c, 64, 64))
        ds_channels = [self._feats[n].shape[1] for n in self.ds_nodes]
        self.train(was_training)
        self.ds_heads = nn.ModuleList([
            nn.Conv2d(c, class_count, kernel_size=3, padding=1) for c in ds_channels
        ])

    def forward(self, x,side=None):
        main = self.model(x)              
        size = x.shape[-2:]
        outs = []
        for n, head in zip(self.ds_nodes, self.ds_heads):
            o = head(self._feats[n])
            outs = [o] + outs
        outs = [main] + outs            
        return outs

class DeepLabV3Plus(nn.Module):
    def __init__(self, args):
        class_count = args["class_count"]
        in_c = args["in_c"]
        super(DeepLabV3Plus, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_c,
            classes=class_count,
            activation=None,
            encoder_output_stride = 8
        )
    def forward(self,x,side=None):
        return [self.model(x)]

if __name__ == "__main__":
    model = UnetPlPl(args={
        "class_count" : 6,
        "in_c" : 3
    })

    ls = torch.zeros(1,3,512,512).float()
    outs = model(ls)
    # print(outs[0].shape)
    # print(model)
    for out in outs :
        print(out.shape)