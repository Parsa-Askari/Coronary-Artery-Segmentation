import torchvision.models as models
import torch.nn as nn
import torch
class Resnet50Side(nn.Module):
    def __init__(self,args):
        super(Resnet50Side,self).__init__()
        model_args = args["resent50"]
        model = models.resnet50(weights='IMAGENET1K_V1')
        model = list(model.children())
        self.backbone = nn.Sequential(*model[:-1])
        self.fnc = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128,2)
        )

    def forward(self,x):
        z = self.backbone(x)
        z = self.fnc(z)
        return z
if __name__=="__main__":
    model = Resnet50Side(args={"resent50":1}).to("cuda")
    x = torch.rand((2,3,512,512)).to("cuda")
    print(model(x).shape)

