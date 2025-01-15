from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
from dinosaur.spherical_harmonic import Grid

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