import gcsfs
import xarray

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
from dinosaur.spherical_harmonic import Grid

def data_generation(path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3', 
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
    
    gcs = gcsfs.GCSFileSystem(token='anon')
    
    full_era5 = xarray.open_zarr(gcs.get_mapper(path), chunks=None)

    full_era5 = full_era5[input_variables + forcing_variables]

    # Creating a new Grid object with custom parameters
    custom_grid = Grid(
        longitude_nodes= longitude_nodes,  # Number of grid points in longitude
        latitude_nodes= latitude_nodes,   # Number of grid points in latitude
    )

    full_era5_grid = spherical_harmonic.Grid(
        latitude_nodes=full_era5.sizes['latitude'],
        longitude_nodes=full_era5.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
    )

    regridder = horizontal_interpolation.ConservativeRegridder(
        full_era5_grid, custom_grid, skipna=True
    )

    sliced_era5 = full_era5.sel(time='2020-01-01T00').compute()
    regridded = xarray_utils.regrid(sliced_era5, regridder)
    regridded = xarray_utils.fill_nan_with_nearest(regridded)
    regridded = regridded.isel(level=level)
    
    return regridded, sliced_era5
    