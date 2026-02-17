import torch 
import torch.nn as nn
import numpy as np
import time

class EncoderBlock(nn.Module):
    def __init__(self,in_c,out_c,k=3,p=1,s=1,downsampling="maxpool",normaliztion="BatchNorm2d",activation="LeakyReLU"):
        super(EncoderBlock,self).__init__()
        activation = getattr(nn, activation)
        normaliztion = getattr(nn,normaliztion)

        self.conv1 = nn.Conv2d(in_channels=in_c,out_channels=out_c,
                               kernel_size=k,padding=p,stride=s)
        self.norm1 = normaliztion(num_features=out_c)
        self.act1 = activation(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_c,out_channels=out_c,
                               kernel_size=k,padding=p,stride=s)
        self.norm2 = normaliztion(num_features=out_c)
        self.act2 = activation(inplace=True)

        
        if(downsampling=="maxpool"):
            self.downsampling = nn.MaxPool2d(kernel_size=2,stride=2)
        else:
            self.downsampling = nn.Conv2d(in_channels=out_c,out_channels=out_c,kernel_size=2,stride=2)
    def forward(self,x):
        """
        inputs : x (B,C,H,W)
        output : z (B,C',H//2,W//2)
        
        """
        z = self.conv1(x)
        z = self.norm1(z)
        z = self.act1(z)

        z = self.conv2(z)
        z = self.norm2(z)
        z = self.act2(z)
        
        out = self.downsampling(z)
        return out,z

class DecoderBlock(nn.Module):
    def __init__(self,x_c,skip_c,out_c,class_count,deep_super_vision,
                 k=3,p=1,s=1,normaliztion="BatchNorm2d",activation="LeakyReLU"):
        super(DecoderBlock,self).__init__()
        activation = getattr(nn, activation)
        normaliztion = getattr(nn,normaliztion)

        self.upsample = nn.ConvTranspose2d(in_channels=x_c,out_channels=x_c//2,
                                           kernel_size=2,stride=2)

        self.conv1 = nn.Conv2d(in_channels=(x_c//2)+skip_c , out_channels = out_c,
                               kernel_size=k,padding=p)
        self.norm1 = normaliztion(num_features=out_c)
        self.act1 = activation(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_c , out_channels = out_c,
                               kernel_size=k,padding=p)
        self.norm2 = normaliztion(num_features=out_c)
        self.act2 = activation(inplace=True)
        
        if(deep_super_vision):
            self.dsv_block = nn.Conv2d(
                in_channels=out_c, 
                out_channels=class_count, 
                kernel_size=1
            )
        self.deep_super_vision = deep_super_vision
    def forward(self,x,skip):
        """
        inputs : x , skip
            - x : (B,C,H,W)
            - skip : (B,C//2,2*H,2*W)
            - pred_embedings : (B,film_hidden_dim)
        outputs : z (B,C//2,H,W)
        """
        u_x = self.upsample(x) # B,C//2,H,W

        z = torch.cat([u_x,skip],dim=1) # B,C,H,W

        z = self.conv1(z)
        z = self.norm1(z)
        z = self.act1(z)

        z = self.conv2(z)
        z = self.norm2(z)
        z = self.act2(z)

        if(self.deep_super_vision):
            dsv_masks = self.dsv_block(z)
            return z , dsv_masks
        return z
class Encoder(nn.Module):
    def __init__(self,channel_list,block_counts=4,
                 k=3,p=1,s=1,downsampling="maxpool",
                 normaliztion="BatchNorm2d",activation="LeakyReLU"):
        super(Encoder,self).__init__()
        layers = []
        for i in range(block_counts):
            in_c = channel_list[i]
            out_c = channel_list[i+1]
            layers+=[
                EncoderBlock(
                    in_c=in_c,
                    out_c=out_c,
                    k=k,
                    p=p,
                    s=s,
                    downsampling=downsampling,
                    normaliztion=normaliztion,
                    activation=activation
                )
            ]
        self.layers = nn.ParameterList(layers)
    def forward(self,x):
        outputs = []
        inp = x
        for layer in self.layers:
            inp , skip = layer(inp)
            outputs.append(skip)

        outputs.reverse() # from the bottom to top
        return outputs , inp
    
class Decoder(nn.Module):
    def __init__(self,channel_list,deep_super_vision,class_count,
                 k=3,p=1,s=1,
                 normaliztion="BatchNorm2d",activation="LeakyReLU"):
        super(Decoder,self).__init__()
        layers = []
        for i,(x_c , skip_c , out_c) in enumerate(channel_list):
            layers+=[
                DecoderBlock(
                    x_c = x_c,
                    skip_c = skip_c,
                    out_c = out_c,
                    k = k,
                    p = p,
                    s = s,
                    normaliztion = normaliztion,
                    activation = activation,
                    class_count = class_count,
                    deep_super_vision = deep_super_vision if i!=len(channel_list)-1 else False
                )
            ]
            self.layers = nn.ParameterList(layers)
    def forward(self,x,skips):
        inp = x
        masks = []
        for i , layer in enumerate(self.layers):
            skip = skips[i]
            outs = layer(inp,skip)
            if(layer.deep_super_vision):
                out , dsv_features = outs
                masks.append(dsv_features)
            else:
                out = outs
            inp = out

        return masks , out
class BottleNeck(nn.Module):
    def __init__(self,in_c,out_c,normaliztion,activation,dropout):
        super(BottleNeck,self).__init__()
        activation = getattr(nn, activation)
        normaliztion = getattr(nn,normaliztion)
        layers1 = [
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                padding=1
            ),
            normaliztion(num_features = out_c),
            activation(inplace=True),
        ]
        layers2 = [
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=3,
                padding=1
            ),
            normaliztion(num_features = out_c),
            activation(inplace=True)

        ]
        if(dropout):
            dropout1 = nn.Dropout2d(p=0.2)
            dropout2 = nn.Dropout2d(p=0.2)
            layers1.append(dropout1)
            layers2.append(dropout2)
        
        layers = layers1+layers2
        self.layers = nn.Sequential(
            *layers
        )
        
    def forward(self,x):
        return self.layers(x)
class MemoryUnet(nn.Module):
    def __init__(self,args):
        super(MemoryUnet,self).__init__()
        model_args = args["memory_unet"]

        layer_count = model_args["layer_count"]
        channel_max = model_args["channel_max"]
        downsampling = model_args["downsampling"]
        normaliztion = model_args["normaliztion"]
        activation = model_args["activation"]
        dropout = model_args["dropout"]
        deep_super_vision = model_args["deep_super_vision"]
        first_layer_out_c = model_args["first_layer_out_c"]

        class_count = args["class_count"]
        in_c = args["in_c"]


        self.encoder_channel_list = [in_c]
        self.decoder_channel_list = []
        # Make the Encoder channel lists 
        for i in range(layer_count):
            if(first_layer_out_c>channel_max):
                first_layer_out_c = channel_max
            self.encoder_channel_list.append(first_layer_out_c)
            first_layer_out_c*=2
        
        # Make the Decoder channel lists 

        for i in range(len(self.encoder_channel_list)-1,0,-1):
            x_c = self.encoder_channel_list[i]*2
            skip_c = self.encoder_channel_list[i]
            if(i==1):
                out_c = self.encoder_channel_list[i]
            else:
                out_c = self.encoder_channel_list[i-1]*2
            self.decoder_channel_list+=[[x_c,skip_c,out_c]]


        self.encoder = Encoder(
            channel_list=self.encoder_channel_list,
            block_counts=layer_count,
            downsampling=downsampling,
            normaliztion=normaliztion,
            activation=activation
        )

        self.bottle_neck = BottleNeck(
            in_c=self.encoder_channel_list[-1],
            out_c=self.encoder_channel_list[-1]*2,
            normaliztion=normaliztion,
            activation=activation,
            dropout = dropout
        )


        self.decoder = Decoder(
            channel_list=self.decoder_channel_list,
            normaliztion=normaliztion,
            activation=activation,
            deep_super_vision = deep_super_vision,
            class_count = class_count
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder_channel_list[1],
                out_channels=class_count,
                kernel_size=1
            )
        )

        self.deep_super_vision = deep_super_vision
        print("encoder info : ")
        print(self.encoder_channel_list)
        print("-----------------------")
    def forward(self,imgs):
        """
            inputs : 
                - imgs : (B,C,H,W) 
            outputs : 
                - preds (B,class_count,H,W)
        """

        skips , b_inp= self.encoder(imgs)

        features = self.bottle_neck(b_inp)

        masks,out = self.decoder(skips = skips , x = features )
        
        pred_mask_logits = self.segmentation_head(out)

        masks += [pred_mask_logits]

        masks.reverse()
        return masks
    
if __name__ == "__main__":
    args = {
        "memory_unet":{
            "layer_count":4,
            "channel_max":256,
            "downsampling":"maxpool",
            "dropout" : False,
            "normaliztion":"InstanceNorm2d",#InstanceNorm2d
            "activation" : "LeakyReLU",
            "first_layer_out_c" : 64,
            "deep_super_vision":True
        },
        "class_count":26,
        "in_c":1
    }
    x = torch.rand((8,1,448,448)).to("cuda")

    model = MemoryUnet(args=args).to("cuda")

    with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
        preds = model(
            imgs = x,
        )
    # time.sleep(5)
    for pred in preds : 
        print(pred.shape)
