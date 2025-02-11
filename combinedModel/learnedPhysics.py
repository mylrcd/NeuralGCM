import torch
import torch.nn as nn
import scipy

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
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, 32)
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
class EPDNetwork(nn.Module):
    def __init__(self, vertical_embedding_size, surface_embedding_size, latent_size=384, num_process_blocks=5, hidden_units=384):
        super(EPDNetwork, self).__init__()

        self.encode = nn.Linear(vertical_embedding_size + surface_embedding_size, latent_size)

        self.process_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_size, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, latent_size)
            )
            for _ in range(num_process_blocks)
        ])

        # Decode : Couche linéaire pour décoder le vecteur latent en sortie
        self.decode = nn.Linear(latent_size, 6)
    def forward(self, vertical_embedding, surface_embedding):
        # Concaténer les embeddings verticaux et de surface
        x = torch.cat((vertical_embedding, surface_embedding), dim=-1)
        
        # Encoder : projeté vers l'espace latent
        x = self.encode(x)
        for block in self.process_blocks:
            x_residual = x  
            x = block(x)
            x += x_residual 


        output = self.decode(x)
        return output
    
    
    
def generate_data(shape):
    data = torch.normal(mean=0, std=1, size=shape).numpy()
    data = scipy.ndimage.gaussian_filter(data, sigma=2)
    return torch.tensor(data)

def generate_cover(level_nodes, longitude_nodes, latitude_nodes):
    sea_ice_cover = torch.empty((level_nodes, longitude_nodes, latitude_nodes))
    sea_surface_temperature = torch.empty((level_nodes, longitude_nodes, latitude_nodes))
    for i in range(level_nodes):
        sea_ice_cover[i] = generate_data((longitude_nodes, latitude_nodes))
        sea_surface_temperature[i] = generate_data((longitude_nodes, latitude_nodes))
    return sea_ice_cover, sea_surface_temperature

def extract_features(regridded_encode) :
    vorticity = torch.tensor(regridded_encode.vorticity.values).permute(0, 1, 2)
    divergence = torch.tensor(regridded_encode.divergence.values).permute(0, 1, 2)
    u_component_of_wind = torch.tensor(regridded_encode.u_component_of_wind.values)#.permute(1, 0, 2, 3)
    v_component_of_wind = torch.tensor(regridded_encode.v_component_of_wind.values)#.permute(1, 0, 2, 3)
    specific_cloud_water = torch.tensor(regridded_encode.specific_cloud_liquid_water_content.values)#.permute(1, 0, 2, 3)
    specific_cloud_ice = torch.tensor(regridded_encode.specific_cloud_ice_water_content.values)#.permute(1, 0, 2, 3)
    specific_humidity = torch.tensor(regridded_encode.specific_humidity.values)#.permute(1, 0, 2, 3)
    temperature_deviation = torch.tensor(regridded_encode.temperature_deviation.values).permute(0, 1, 2)
    
    land_fraction = torch.randn(32 * 64 * 32, 1)
    sea_fraction = torch.randn(32 * 64 * 32, 1)
    ice_fraction = torch.randn(32 * 64 * 32, 1)
    
    sea_ice_cover, sea_surface_temperature = generate_cover(32, 64, 32)

    return vorticity, divergence, u_component_of_wind, v_component_of_wind, specific_cloud_water, specific_cloud_ice, specific_humidity, temperature_deviation, land_fraction, sea_fraction, ice_fraction, sea_ice_cover, sea_surface_temperature

def stack_features(regridded_encode) :
    vorticity, divergence, u_component_of_wind, v_component_of_wind, specific_cloud_water, specific_cloud_ice, specific_humidity, temperature_deviation, land_fraction, sea_fraction, ice_fraction, sea_ice_cover, sea_surface_temperature = extract_features(regridded_encode)
    features = torch.stack((vorticity, divergence, u_component_of_wind, v_component_of_wind, specific_cloud_water, specific_cloud_ice, specific_humidity, temperature_deviation), dim=1)

    features = features.reshape(-1, 8, 32) 
    
    specific_humidity_init = specific_humidity
    temperature_deviation_init = temperature_deviation

    sea_input = sea_surface_temperature
    ice_input = sea_ice_cover 
    
    sea_input = sea_input.permute(1, 2, 0)
    sea_input = sea_input.reshape(-1, 32)
    sea_input = sea_input.reshape(-1, 1)

    ice_input = ice_input.permute(1, 2, 0)
    ice_input = ice_input.reshape(-1, 32)
    ice_input = ice_input.reshape(-1, 1)
    
    
    land_input = torch.stack([temperature_deviation_init, specific_humidity_init])
    land_input = land_input.reshape(-1, 2, 32) 
    land_input = land_input.reshape(-1, 2)
    
    return features, sea_input, ice_input, land_input, land_fraction, sea_fraction, ice_fraction, sea_ice_cover, sea_surface_temperature


def input_time(regridded_encode) :
    features, sea_input, ice_input, land_input, land_fraction, sea_fraction, ice_fraction, sea_ice_cover, sea_surface_temperature = stack_features(regridded_encode)

    features_time = features
    land_input_time = land_input

    features_time = features
    
    return features_time, sea_input, ice_input, land_input_time, land_fraction, sea_fraction, ice_fraction
class LearnedPhysicsModel(nn.Module):
    def __init__(self, vertical_size=32, surface_size=8):
        super(LearnedPhysicsModel, self).__init__()
        self.vertical_net = VerticalEmbeddingNetwork()
        self.sea_net    = SurfaceEmbeddingNetwork(surface_type='sea')
        self.land_net   = SurfaceEmbeddingNetwork(surface_type='land')
        self.ice_net    = SurfaceEmbeddingNetwork(surface_type='ice')
        self.area_sum   = AreaWeightedSum()
        self.epd_net    = EPDNetwork(vertical_size, surface_size)

    def forward(self, regridded_encode):
        features_time, sea_in, ice_in, land_in, land_frac, sea_frac, ice_frac = input_time(regridded_encode)
        vertical_emb = self.vertical_net(features_time)
        sea_emb      = self.sea_net(sea_in)
        land_emb     = self.land_net(land_in)
        ice_emb      = self.ice_net(ice_in)
        surface_emb  = self.area_sum(land_emb, sea_emb, ice_emb, land_frac, sea_frac, ice_frac)
        return self.epd_net(vertical_emb, surface_emb)