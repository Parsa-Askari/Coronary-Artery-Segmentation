import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
###IE###
###SS###
class ConvBlock(nn.Module):
    def __init__(self,in_c , out_c,s,k=3,p=1):
        super(ConvBlock,self).__init__()
        self.layers =  nn.Sequential( 
            nn.Conv2d(
                in_channels = in_c , 
                out_channels = out_c ,
                kernel_size=k, 
                stride = s ,
                padding = p
            ),
            nn.InstanceNorm2d(out_c, eps=1e-5, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
    def forward(self,x):
        return self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self,in_c,out_c,init_stride):
        super(EncoderBlock,self).__init__()
        self.conv1 = ConvBlock(in_c=in_c,out_c=out_c,s=init_stride)
        self.conv2 = ConvBlock(in_c=out_c,out_c=out_c,s=1)
    def forward(self,x):
        z1 = self.conv1(x)
        z2 = self.conv2(z1)
        return z2

class DecoderBlock(nn.Module):
    def __init__(self,skip_c , prev_c):
        super(DecoderBlock,self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=prev_c,out_channels=skip_c,stride=2,kernel_size=2)
        self.conv1 = ConvBlock(in_c=skip_c*2 , out_c=skip_c,s=1)
        self.conv2 = ConvBlock(in_c=skip_c,out_c=skip_c,s=1)
    def forward(self,prev , skip):
        up_z = self.upsample(prev)
        if(up_z.shape[2] != skip.shape[2] or up_z.shape[3] != skip.shape[3]):
            up_z = F.interpolate(
                up_z, 
                size=skip.shape[2:], 
                mode="bilinear", 
                align_corners=False
            )
        cat_z = torch.concat([skip,up_z],dim=1)
        z1 = self.conv1(cat_z)
        z2 = self.conv2(z1)
        return z2

class Encoder(nn.Module):
    def __init__(self,encoder_settings,stride_settings,input_channels):
        super(Encoder,self).__init__()
        encoder_layers = []
        in_c = input_channels
        for i in range(len(encoder_settings)):
            s = stride_settings[i]
            out_c = encoder_settings[i]
            encoder_layers.append(EncoderBlock(in_c=in_c,out_c=out_c,init_stride=s))
            in_c = out_c
        self.encoder_layers = nn.ModuleList(encoder_layers)
    def forward(self,x):
        skip_features = []
        in_feature = x
        for layer in self.encoder_layers:
            out_feature = layer(in_feature)
            skip_features.append(out_feature)
            in_feature = out_feature
        return skip_features
    

class Decoder(nn.Module):
    def __init__(self,encoder_settings,deep_super_vision,class_count=None):
        super(Decoder,self).__init__()
        decoder_layers = []
        if(deep_super_vision):
            dsv_layers= []
        for i in range(len(encoder_settings)-1,0,-1):
            prev_c = encoder_settings[i]
            skip_c = encoder_settings[i-1]
            decoder_layers.append(DecoderBlock(skip_c=skip_c,prev_c=prev_c))
            if(deep_super_vision):
                dsv_layers.append(nn.Conv2d(in_channels=skip_c,out_channels=class_count,kernel_size=1))
        self.decoder_layers = nn.ModuleList(decoder_layers)
        if(deep_super_vision):
            self.dsv_layers = nn.ModuleList(dsv_layers)
        else:
            self.dsv_layers = nn.ModuleList([nn.Conv2d(in_channels=skip_c,out_channels=class_count,kernel_size=1)])

        self.deep_super_vision = deep_super_vision
    def forward(self,skip_features):
        prev = skip_features[-1]
        out_features = []
        for i,layer in enumerate(self.decoder_layers):
            skip = skip_features[-(i)-2] 
            prev = layer(prev,skip)
            if(self.deep_super_vision):
                dsv_layer = self.dsv_layers[i]
                out = dsv_layer(prev)
                out_features.append(out)
            elif(i==len(self.decoder_layers)-1):
                dsv_layer = self.dsv_layers[0]
                out = dsv_layer(prev)
                out_features.append(out)
                
        return out_features
    

class nnUnetv2(nn.Module):
    def __init__(self,args):
        super(nnUnetv2,self).__init__()
        class_count = args["class_count"]
        attention = args["attention"]
        image_shape = args["image_shape"]
        base_channel = args["base_channel"]
        f_int_scale = args["f_int_scale"]
        max_channels = args["max_channels"]
        input_channels = args["input_channels"]
        self.deep_super_vision = args["deep_super_vision"]
        h = image_shape[0]
        w = image_shape[1]

        stage_count = 1
        c = base_channel
        encoder_settings = []

        while(h>4 and w>4):
            stage_count+=1
            h//=2
            w//=2
            c*=2
            if(c>max_channels):
                c = max_channels
            encoder_settings.append(c)
        
        encoder_settings = [base_channel] + encoder_settings
        stride_settings = [2 if i !=0 else 1 for i in range(len(encoder_settings))]
        self.encoder = Encoder(
            encoder_settings = encoder_settings,
            stride_settings = stride_settings,
            input_channels = input_channels
        )

        self.decoder = Decoder(
            encoder_settings=encoder_settings,
            deep_super_vision = self.deep_super_vision,
            class_count = class_count
        )
        print("encoder settings : " , encoder_settings)
        print("stride settings : " ,stride_settings)
        print("stage count : "  , stage_count)
    def forward(self,x):
        skip_features = self.encoder(x)
        out_features = self.decoder(skip_features)
        out_features.reverse()
        return out_features
        
if __name__ == "__main__":
    args = {
        "base_path" : "../arcade/nnUnet_dataset/syntax",
        "in_c" : 1,
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
        "epcohs":30,
        "f_int_scale" : 2,
        "full_report_cycle" : 10,
        "max_channels":512,
        "input_channels":1,
        "loss_type":"dice loss",
        "alpha":0.75,
        "beta":0.25,
        "gamma":1.00,
        "f_gamma":2.0,
        "f_loss_scale":1,
        "loss_coefs":{"CE":1.0,"Second":1.0},
        "output_base_path" : "./outputs",
        "name" : "Attention7-AllClass",
        "deep_super_vision" : True
    }
    class_map = {
        1: '1',2: '2', 3: '3',4: '4',
        5: '5',6: '6',7: '7',8: '8',
        9: '9',10: '9a',11: '10',12: '10a',
        13: '11',14: '12',15: '12a',16: '13',
        17: '14',18: '14a',19: '15',20: '16',
        21: '16a',22: '16b',23: '16c',
        24: '12b',25: '14b'
    }
    model = nnUnetv2(args)
    ls = torch.ones((2,1,448,448)).float()
    outs = model(ls)
    for out in outs:
        print(out.shape)

    # """
    # torch.Size([10, 32, 256, 256])
    # torch.Size([10, 64, 128, 128])
    # torch.Size([10, 128, 64, 64])
    # torch.Size([10, 256, 32, 32])
    # torch.Size([10, 512, 16, 16])
    # torch.Size([10, 512, 8, 8])
    # torch.Size([10, 512, 4, 4])
    # """