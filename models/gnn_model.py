import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv ,SAGEConv, GraphNorm
import torch.nn.functional as F
from torch_geometric.utils import subgraph
###IE###
from utils.helpers import load_model
###SS###
class SimpleGCN(nn.Module):
    def __init__(self, args):
        super(SimpleGCN,self).__init__()
        model_args = args["gnn_simple"]
        hidden_size = model_args["hidden_channels"]
        class_count = args["class_count"]
        feature_size = args["feature_size"]

        self.conv1 = GCNConv(feature_size,64)
        self.bn1 = nn.BatchNorm1d(64) 
        
        self.conv2 = GCNConv(64,64)
        self.bn2 = nn.BatchNorm1d(64) 
        
        self.conv3 = GCNConv(64,32)
        self.bn3 = nn.BatchNorm1d(32) 
        
        self.classifier = nn.Linear(32,class_count)
        self.relu = nn.LeakyReLU()
    def forward(self,x,edge_index,batch=None):
        z = self.conv1(x,edge_index)
        z = self.bn1(z)  
        z = self.relu(z)

        z = self.conv2(z,edge_index)
        z = self.bn2(z)  
        z = self.relu(z)

        z = self.conv3(z,edge_index)
        z = self.bn3(z)
        z = self.relu(z)

        z = self.classifier(z)
        return z


class VesselGraphSAGE(nn.Module):
    def __init__(self, args):
        super().__init__()

        in_channels = args["feature_size"]
        hidden = args["gnn_simple"].get("hidden_channels", 128)
        class_count = args["class_count"]
        dropout = args["gnn_simple"].get("dropout", 0.25)


        self.conv1 = SAGEConv(in_channels, hidden)
        self.norm1 = GraphNorm(hidden)

        self.conv2 = SAGEConv(hidden, hidden)
        self.norm2 = GraphNorm(hidden)

        self.conv3 = SAGEConv(hidden, hidden)
        self.norm3 = GraphNorm(hidden)

        self.conv4 = SAGEConv(hidden, hidden)
        self.norm4 = GraphNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden, class_count)
        )

        self.dropout = dropout
    
    
    def forward(self, x, edge_index, batch=None):

        z = self.conv1(x, edge_index)
        z = self.norm1(z, batch)
        z = F.leaky_relu(z, 0.1)
        z = F.dropout(z, p=self.dropout, training=self.training)

        z_res = z
        z = self.conv2(z, edge_index)
        z = self.norm2(z, batch)
        z = F.leaky_relu(z, 0.1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = z + z_res

        z_res = z
        z = self.conv3(z, edge_index)
        z = self.norm3(z, batch)
        z = F.leaky_relu(z, 0.1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = z + z_res

        z_res = z
        z = self.conv4(z, edge_index)
        z = self.norm4(z, batch)
        z = F.leaky_relu(z, 0.1)
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = z + z_res

        z = self.classifier(z)
        

        return z


class GatGNN(torch.nn.Module):
    def __init__(self,args):
        super(GatGNN, self).__init__()
        model_args = args["gat_simple"]
        feature_size = args["feature_size"]
        hidden_channels = model_args["hidden_channels"]
        class_count = args["class_count"]

        self.conv1 = GATConv(feature_size, hidden_channels // 4, heads=4)
        self.bn1 = nn.BatchNorm1d(hidden_channels) 

        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4)
        self.bn2 = nn.BatchNorm1d(hidden_channels) 
        
        self.conv3 = GATConv(hidden_channels, 10, heads=1)
        self.bn3 = nn.BatchNorm1d(10) 

        self.classifier = nn.Linear(10,class_count)
    def forward(self,x,edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        out = self.classifier(x3)
        return out 



if __name__ == "__main__":
    x = torch.tensor([
        [0.2,0.3],[0.4,0.5]
    ]).to("cuda")
    edge_index = torch.tensor([[0,1],[1,0]]).to("cuda")
    model = SimpleGCN(args={"gnn_simple":{"hidden_size":64},"class_count":20}).to("cuda")
    pred = model(x,edge_index)
    print(pred)
    