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
    def __init__(self,x_c,skip_c,out_c,film_hidden_dim,
                 k=3,p=1,s=1,normaliztion="BatchNorm2d",activation="LeakyReLU"):
        super(DecoderBlock,self).__init__()
        activation = getattr(nn, activation)
        normaliztion = getattr(nn,normaliztion)

        self.upsample = nn.ConvTranspose2d(in_channels=x_c,out_channels=x_c//2,
                                           kernel_size=2,stride=2)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=film_hidden_dim,out_features=x_c)
        )
        self.conv1 = nn.Conv2d(in_channels=(x_c//2)+skip_c , out_channels = out_c,
                               kernel_size=3,padding=1)
        self.norm1 = normaliztion(num_features=out_c)
        self.act1 = activation(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_c , out_channels = out_c,
                               kernel_size=3,padding=1)
        self.norm2 = normaliztion(num_features=out_c)
        self.act2 = activation(inplace=True)
        
    def forward(self,x,skip,pred_embedings):
        """
        inputs : x , skip
            - x : (B,C,H,W)
            - skip : (B,C//2,2*H,2*W)
            - pred_embedings : (B,film_hidden_dim)
        outputs : z (B,C//2,H,W)
        """
        u_x = self.upsample(x) # B,C//2,H,W
        layer_condition = self.mlp(pred_embedings).unsqueeze(-1).unsqueeze(-1) # B,C,1,1
        # print(layer_condition.shape)
        # print(u_x.shape)
        # print(layer_condition[:,:u_x.shape[1]].shape)
        u_x = u_x*layer_condition[:,:u_x.shape[1]] + layer_condition[:,u_x.shape[1]:] # B,C//2,H,W

        z = torch.cat([u_x,skip],dim=1) # B,C,H,W

        z = self.conv1(z)
        z = self.norm1(z)
        z = self.act1(z)

        z = self.conv2(z)
        z = self.norm2(z)
        z = self.act2(z)

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
    def __init__(self,channel_list,film_hidden_dim,
                 k=3,p=1,s=1,
                 normaliztion="BatchNorm2d",activation="LeakyReLU"):
        super(Decoder,self).__init__()
        layers = []
        for x_c , skip_c , out_c in channel_list:
            layers+=[
                DecoderBlock(
                    x_c = x_c,
                    skip_c = skip_c,
                    out_c = out_c,
                    film_hidden_dim = film_hidden_dim,
                    k = k,
                    p = p,
                    s = s,
                    normaliztion = normaliztion,
                    activation = activation
                )
            ]
            self.layers = nn.ParameterList(layers)

    def forward(self,x,skips,pred_embedings):
        inp = x
        for i , layer in enumerate(self.layers):
            skip = skips[i]
            out = layer(inp,skip,pred_embedings)
            inp = out

        return out
class BottleNeck(nn.Module):
    def __init__(self,in_c,out_c,normaliztion,activation):
        super(BottleNeck,self).__init__()
        activation = getattr(nn, activation)
        normaliztion = getattr(nn,normaliztion)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                padding=1
            ),
            normaliztion(num_features = out_c),
            activation(inplace=True),
            
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=3,
                padding=1
            ),
            normaliztion(num_features = out_c),
            activation(inplace=True)
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
        embeding_dim = model_args["embeding_dim"]
        film_hidden_dim = model_args["film_hidden_dim"]
        use_multi_class_mask = model_args["use_multi_class_mask"]

        activation = model_args["activation"]
        first_layer_out_c = model_args["first_layer_out_c"]

        class_count = args["class_count"]
        in_c = args["in_c"]

        if(use_multi_class_mask):
            in_c += embeding_dim
        self.encoder_channel_list = [in_c+1]
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
        
        if(use_multi_class_mask):
            self.mask_embeding_layer = nn.Embedding(class_count,embeding_dim)

        self.q_emebeding_layer = nn.Embedding(class_count,embeding_dim)

        self.GAP = nn.AdaptiveMaxPool2d((1,1))

        self.v_seen_mlp = nn.Sequential(
            nn.Linear(in_features=class_count,out_features=embeding_dim),
            nn.ReLU(inplace=True)
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(
                in_features=self.encoder_channel_list[-1]*2+embeding_dim+1,
                out_features=self.encoder_channel_list[-1]*2+embeding_dim+1
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.encoder_channel_list[-1]*2+embeding_dim+1,
                out_features=class_count
            ),
        )


        self.film_base_mlp = nn.Sequential(
            nn.Linear(in_features=embeding_dim,out_features=film_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=film_hidden_dim , out_features=film_hidden_dim),
            nn.ReLU(inplace=True)
        )
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
            activation=activation
        )


        self.decoder = Decoder(
            channel_list=self.decoder_channel_list,
            normaliztion=normaliztion,
            activation=activation,
            film_hidden_dim = film_hidden_dim
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder_channel_list[1],
                out_channels=class_count,
                kernel_size=1
            )
        )
        print("encoder info : ")
        print(self.encoder_channel_list)
        print("-----------------------")
    def forward(self,img,m_taken,v_seen,m_labels=None):
        """
            inputs : 
                - img : TF (BxT,1,H,W) | (B,1,H,W) 
                - m_taken : TF (BxT,1,H,W) | (B,1,H,W)
                - m_labels : TF (BxT,1,H,W) | (B,1,H,W)
                - v_seen : (BxT,26) | (B,26)
            outputs : 
                - pred_mask_logits : TF (BxT,C,H,W) | (B,C,H,W)
                - label_logits : TF (BxT,26) | (B,26)
                - pred_label : TF (BxT) | (B)
        """

        print(img.shape,m_taken.shape,v_seen.shape)
        enc_inp = [img,m_taken]
        if(m_labels is not None):
            emb_labels = self.mask_embeding_layer(m_labels)
            emb_labels = emb_labels.permute(0, 3, 1, 2)
            enc_inp.append(emb_labels)

        enc_inp = torch.cat(enc_inp,dim=1) # (BxT,C,H,W)

        skips , b_inp= self.encoder(enc_inp)

        features = self.bottle_neck(b_inp)
        ## Controller 
        gap_f =self.GAP(features).reshape(features.shape[0],features.shape[1]) # (BxT,D)
        v_seen_t = self.v_seen_mlp(v_seen) # (BxT,embed_size)
        fraction_taken = m_taken.mean(dim=(2,3)) # (BxT,1)
        label_logits = self.condition_mlp(
            torch.cat([gap_f,v_seen_t,fraction_taken],dim=1)
        )# (BxT,26)
        
        pred_label = torch.argmax(label_logits,dim=1) #(BxT)

        ## FILM
        pred_embedings = self.q_emebeding_layer(pred_label)
        pred_embedings = self.film_base_mlp(pred_embedings)

        out = self.decoder(skips = skips , x = features , pred_embedings = pred_embedings)
        
        pred_mask_logits = self.segmentation_head(out)
        # print(out.shape)
        return pred_mask_logits , label_logits, pred_label
    
if __name__ == "__main__":
    args = {
        "memory_unet":{
            "layer_count":4,
            "channel_max":256,
            "downsampling":"maxpool",
            "normaliztion":"InstanceNorm2d",
            "activation" : "LeakyReLU",
            "first_layer_out_c" : 64,
            "embeding_dim":5,
            "film_hidden_dim":128,
            "mini_batch_size":4,
            "use_multi_class_mask": False,
        },
        "class_count":26,
        "in_c":1
    }
    x = torch.rand((14,1,448,448)).to("cuda")
    
    m_taken = torch.zeros((14,1,448,448)).to("cuda")
    m_taken[0,0,10:200,200:250]=1
    m_taken[1,0,200:220,10:200]=1

    m_labels = torch.zeros((14,1,448,448),dtype=torch.long).to("cuda")
    m_labels[0,0,10:200,200:250]=2
    m_labels[1,0,200:220,10:200]=25
    m_labels = m_labels.squeeze(1)

    v_seen = torch.zeros(14,26).to("cuda")
    v_seen[0][1:3] = 1

    model = MemoryUnet(args=args).to("cuda")
    with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=True):
        preds = model(
            img = x,
            m_taken = m_taken,
            # m_labels = m_labels,
            v_seen = v_seen
        )
    time.sleep(5)
    print(preds[0].shape)
    print(preds[1])
