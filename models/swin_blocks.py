import torch.nn as nn
###IE###
###SS###
class CostumeBlock(nn.Module):
    def __init__(self,in_c,out_c):
        super(CostumeBlock,self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=2,
                stride=2,
                bias=True
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.layer(x)
class CostumeHead(nn.Module):
    def __init__(self,input_h,input_w,depth,emb_size,class_count,deep_super_vision):
        super(CostumeHead,self).__init__()
        self.layers = nn.ModuleList()
        self.dsv_layers = nn.ModuleList()

        self.input_w = input_w
        self.input_h = input_h
        self.deep_super_vision = deep_super_vision
        in_c = emb_size*(2**(depth-1))
        for i in range(depth+1):
            out_c = in_c//2
            self.layers.append(CostumeBlock(in_c=in_c,out_c=out_c))
            if(self.deep_super_vision):
                self.dsv_layers.append(nn.Conv2d(in_channels=out_c,out_channels=class_count,kernel_size=1))
            in_c = out_c

        if(len(self.dsv_layers)==0):
            self.dsv_layers.append(nn.Conv2d(in_channels=out_c,out_channels=class_count,kernel_size=1))
        
        scale = 2**(depth+1)
        if(input_h//scale!=int(input_h/scale) or input_w//scale!=int(input_w/scale)):
            self.interpolate = True
        else :
            self.interpolate = False
    def forward(self,x):
        out_features = []
        in_z = x
        for i,layer in enumerate(self.layers):
            out_z = layer(in_z)
            if(self.deep_super_vision):
                deep_z = self.dsv_layers[i](out_z)
                out_features =[deep_z] + out_features
            elif (i==len(self.layers)-1):
                deep_z = self.dsv_layers[-1](out_z)
                out_features =[deep_z] + out_features
            in_z = out_z

        return out_features