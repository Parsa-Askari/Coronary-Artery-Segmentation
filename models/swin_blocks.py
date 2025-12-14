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
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(
                in_channels = out_c , 
                out_channels = out_c ,
                kernel_size=3, 
                stride = 1 ,
                padding = 1
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels = out_c , 
                out_channels = out_c ,
                kernel_size=3, 
                stride = 1 ,
                padding = 1
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.layer(x)
class CostumeHead(nn.Module):
    def __init__(self,input_h,input_w,depth,emb_size,
                 class_count,abs_class_count,deep_super_vision):
        super(CostumeHead,self).__init__()
        self.layers = nn.ModuleList()
        self.dsv_layers = nn.ModuleList()

        self.input_w = input_w
        self.input_h = input_h
        self.deep_super_vision = deep_super_vision
        in_c = emb_size*(2**(depth-1))
        
        self.stage1 = CostumeBlock(in_c=in_c,out_c=in_c//2) # C//2 x H/16 x W/16
        if(self.deep_super_vision):
            self.dsv1 = nn.Conv2d(
                in_channels=in_c//2,
                out_channels=class_count,
                kernel_size=1)
        in_c//=2

        self.stage2 = CostumeBlock(in_c=in_c,out_c=in_c//2) # C//4 x H/8 x W/8
        if(self.deep_super_vision):
            self.dsv2 = nn.Conv2d(
                in_channels=in_c//2,
                out_channels=class_count,
                kernel_size=1)
        in_c//=2

        self.stage3 = CostumeBlock(in_c=in_c,out_c=in_c//2) # C//8 x H/4 x W/4
        if(self.deep_super_vision):
            self.dsv3 = nn.Conv2d(
                in_channels=in_c//2,
                out_channels=class_count,
                kernel_size=1)
        in_c//=2
        
        self.stage4 = CostumeBlock(in_c=in_c,out_c=in_c//2) # C//16 x H/2 x W/2
        if(self.deep_super_vision):
            self.dsv4 = nn.Conv2d(
                in_channels=in_c//2,
                out_channels=class_count,
                kernel_size=1)
        in_c//=2
        

        self.stage5 = CostumeBlock(in_c=in_c,out_c=in_c//2) # C//32 x H x W
        in_c//=2

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c, 
                out_channels=in_c//2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(in_c//2),
            nn.LeakyReLU(),
            

            nn.Conv2d(
                in_channels=in_c//2,
                out_channels=in_c//2,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(in_c//2),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=in_c//2,out_channels=class_count,kernel_size=1)
        )
        
        
    def forward(self,x):
        z = self.stage1(x)

        z1 = self.stage2(z)
        
        z2 = self.stage3(z1)
        z3 = self.stage4(z2)

        z4 = self.stage5(z3)
        z5 = self.head(z4)

        
        if(self.deep_super_vision):
            z_dsv = self.dsv1(z)
            z1_dsv = self.dsv2(z1)
            z2_dsv = self.dsv3(z2)
            z3_dsv = self.dsv4(z3)
            return [z5,z3_dsv,z2_dsv,z1_dsv,z_dsv]
        else:
            return [z5]
        # torch.Size([8, 512, 24, 24]) 
        # torch.Size([8, 256, 48, 48]) 
        # torch.Size([8, 128, 96, 96]) 
        # torch.Size([8, 64, 192, 192]) 
        # torch.Size([8, 32, 384, 384])