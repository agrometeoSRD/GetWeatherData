import os
import xarray as xr
from clisops.core import subset

shape_path = f"C:\\Users\\{os.getenv('USERNAME')}\\OneDrive - IRDA\\GIS\\RegionAgricolesQC.geojson"

def subset_by_shape(ds: xr.Dataset, shape_path: str) -> xr.Dataset:
    """
    Subset the given dataset by a shapefile.

    Parameters:
    - ds: Input xarray Dataset
    - shape_path: Path to the shapefile (.geojson)

    Returns:
    An xarray Dataset subsetted by the shape
    """
    return subset.subset_shape(ds, shape=shape_path)