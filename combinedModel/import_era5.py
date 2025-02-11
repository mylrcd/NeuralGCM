from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
from dinosaur.spherical_harmonic import Grid

import xarray as xr
import numpy as np
import jax
import gcsfs
import pickle

import sys
sys.path.append("c:/Users/mayeu/SynologyDrive/DossiersMayeul/Mes Documents/TSP/3A/PFE/NeuralGCM/combinedModel/neuralgcmPFE")
from neuralgcmPFE import neuralgcm

model_name = 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl'  #@param ['neural_gcm_dynamic_forcing_deterministic_0_7_deg.pkl', 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl', 'neural_gcm_dynamic_forcing_deterministic_2_8_deg.pkl', 'neural_gcm_dynamic_forcing_stochastic_1_4_deg.pkl'] {type: "string"}

rng_key = jax.random.PRNGKey(42)


input_variables =['geopotential', 
                    'specific_humidity','temperature',
                    'u_component_of_wind',
                    'v_component_of_wind',
                    'specific_cloud_ice_water_content',
                    'specific_cloud_liquid_water_content']

forcing_variables = ['sea_ice_cover', 'sea_surface_temperature']

longitude_nodes=64
latitude_nodes=32

longitude_nodes_original = 256
latitude_nodes_original = 128
level_original = [i for i in range(37)]
data_inner_steps = 6

path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

def data_generation(data, 
            demo_start_time = '2020-02-14',
            demo_end_time = '2020-02-18',
            data_inner_steps = 24,  
            input_variables = ['geopotential', 
                                'specific_humidity','temperature',
                                'u_component_of_wind',
                                'v_component_of_wind',
                                'specific_cloud_ice_water_content',
                                'specific_cloud_liquid_water_content'], 
            forcing_variables = ['sea_ice_cover', 'sea_surface_temperature'], 
            longitude_nodes=32, 
            latitude_nodes=16,
            level = [0, 10, 20, 30, 36]):

    sliced_era5 = (
            data
            [input_variables + forcing_variables]
            .pipe(
                xarray_utils.selective_temporal_shift,
                variables=forcing_variables,
                time_shift='24 hours',
            )
            .sel(time=slice(demo_start_time, demo_end_time, data_inner_steps))
            .compute()
        )

    # Creating a new Grid object with custom parameters
    custom_grid = Grid(
        longitude_nodes= longitude_nodes,  # Number of grid points in longitude
        latitude_nodes= latitude_nodes,   # Number of grid points in latitude
    )

    sliced_era5_grid = spherical_harmonic.Grid(
        latitude_nodes=sliced_era5.sizes['latitude'],
        longitude_nodes=sliced_era5.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(sliced_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(sliced_era5.longitude),
    )

    regridder = horizontal_interpolation.ConservativeRegridder(
        sliced_era5_grid, custom_grid, skipna=True
    )

    regridded = xarray_utils.regrid(sliced_era5, regridder)
    regridded = xarray_utils.fill_nan_with_nearest(regridded)
    regridded = regridded.isel(level=level)
    
    return regridded, sliced_era5


def data_rescale(data, 
            longitude_nodes=32, 
            latitude_nodes=16,
            level = [0, 10, 20, 30, 36]):


    # Creating a new Grid object with custom parameters
    custom_grid = Grid(
        longitude_nodes= longitude_nodes,  # Number of grid points in longitude
        latitude_nodes= latitude_nodes,   # Number of grid points in latitude
    )

    data_grid = spherical_harmonic.Grid(
        latitude_nodes=data.sizes['latitude'],
        longitude_nodes=data.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(data.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(data.longitude),
    )

    regridder = horizontal_interpolation.ConservativeRegridder(
        data_grid, custom_grid, skipna=True
    )

    regridded = xarray_utils.regrid(data, regridder)
    regridded = xarray_utils.fill_nan_with_nearest(regridded)
    regridded = regridded.isel(level=level)
    
    return regridded

def generation_xarray_encode(regridded_original, model, longitude_nodes, latitude_nodes, level_nodes) :
    states_divergence = np.empty((5, 32, 256, 128), dtype='float32') 
    states_log_surface_pressure = np.empty((5, 32, 256, 128), dtype='float32') 
    states_vorticity = np.empty((5, 32, 256, 128), dtype='float32') 
    states_temperature_deviation = np.empty((5, 32, 256, 128), dtype='float32') 

    for i in range(5):
        inputs_original = model.inputs_from_xarray(regridded_original.isel(time=i))
        input_forcings_original = model.forcings_from_xarray(regridded_original.isel(time=i))
        state = model.encode(inputs_original, input_forcings_original, rng_key)
        states_divergence[i] = state.state.divergence[ :, :, :128]
        states_log_surface_pressure[i] = state.state.log_surface_pressure[ :, :, :128]
        states_vorticity[i] = state.state.vorticity[ :, :, :128]
        states_temperature_deviation[i] = state.state.temperature_variation[ :, :, :128]
        
    states_divergence = np.transpose(states_divergence, (1, 0, 2, 3))
    states_vorticity = np.transpose(states_vorticity, (1, 0, 2, 3))
    states_temperature_deviation = np.transpose(states_temperature_deviation, (1, 0, 2, 3))
    states_log_surface_pressure = np.transpose(states_log_surface_pressure, (1, 0, 2, 3))
    
    regridded_original_encode = regridded_original.isel(time=slice(0, 5))
    regridded_original_encode = regridded_original_encode.isel(level=slice(0, 32))
    regridded_original_encode = regridded_original_encode.isel(time=slice(0, 5))
    regridded_original_encode['divergence'] = xr.DataArray(states_divergence, coords=regridded_original_encode.coords, dims=regridded_original_encode.dims)
    regridded_original_encode['vorticity'] = xr.DataArray(states_vorticity, coords=regridded_original_encode.coords, dims=regridded_original_encode.dims)
    regridded_original_encode['log_surface_pressure'] = xr.DataArray(states_log_surface_pressure, coords=regridded_original_encode.coords, dims=regridded_original_encode.dims)
    regridded_original_encode['temperature_deviation'] = xr.DataArray(states_temperature_deviation, coords=regridded_original_encode.coords, dims=regridded_original_encode.dims)
    
    regridded_encode = data_rescale(data = regridded_original_encode,
        longitude_nodes = longitude_nodes,
        latitude_nodes = latitude_nodes,
        level = level_nodes)
    
    return regridded_encode


def generate_final_xarray(path, gcs, model, start_time, end_time, data_inner_steps, input_variables, forcing_variables, longitude_nodes_original, latitude_nodes_original, level_original):
    
    full_era5 = xr.open_zarr(gcs.get_mapper(path), chunks=None)
    
    regridded_original, _ = data_generation(data = full_era5,
        demo_start_time = start_time,
        demo_end_time = end_time,
        data_inner_steps = data_inner_steps, 
        input_variables = input_variables,
        forcing_variables = forcing_variables,
        longitude_nodes = longitude_nodes_original,
        latitude_nodes = latitude_nodes_original,
        level = level_original)
    
    regridded_encode = generation_xarray_encode(regridded_original, model, longitude_nodes_original, latitude_nodes_original, level_original)
    

    return regridded_encode


demo_start_time = '2020-02-14'
demo_end_time = '2020-02-18'