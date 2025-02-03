import torch.nn as nn

class VerticalEmbeddingNetwork(nn.Module):
    def __init__(self):
        super(VerticalEmbeddingNetwork, self).__init__()
        
        #we code the convolution network describe in the paper
        self.net = nn.Sequential(
            nn.Conv1d(in_channels = 8, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1),
            )
        
    def forward(self, x) :
        x = self.net(x)
        return x
    
    
class SurfaceEmbeddingNetwork(nn.Module):
    def __init__(self, surface_type):
        super(SurfaceEmbeddingNetwork, self).__init__()
        self.surface_type = surface_type
        
        if self.surface_type == 'sea' :
            self.net = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_features = 1, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 8)
            )
        
        elif self.surface_type == 'land' :
            self.net = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_features = 2, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 8)
            )   
            
        elif self.surface_type == 'ice' :
            self.net = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_features = 1, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 16),
                nn.ReLU(),
                nn.Linear(in_features = 16, out_features = 8)
            )
        
    def forward(self, x) :
        x = self.net(x)
        return x
    
class AreaWeightedSum(nn.Module):
    def __init__(self):
        super(AreaWeightedSum, self).__init__()
    
    def forward(self, sea_embedding, land_embedding, ice_embedding, sea_fraction, land_fraction, ice_fraction):
        return sea_embedding * sea_fraction + land_embedding * land_fraction + ice_embedding * ice_fraction