from pathlib import Path
import xarray as xr
from clisops.core import subset

#todo : try to find a way to use the get_project_root from utils...
def get_project_root(current_directory: Path) -> Path:
    if (current_directory / 'utils').exists():
        return current_directory
    parent_directory = current_directory.parent
    if parent_directory == current_directory:
        raise FileNotFoundError("Failed to find the project root directory.")
    return get_project_root(parent_directory)

def get_shape_filename():
    # get the location of the RegionAgricolesQC geojson file. Located in /data/GIS
    # Start by getting the directory of the root folder
    project_root = get_project_root(Path(__file__).parent)

    # Get the path to the geojson file
    area_filename = 'RegionAgricolesQC.geojson'
    area_path = project_root / 'data/GIS/PAVICS/' / area_filename
    return area_path

def subset_by_shape(ds: xr.Dataset, area_path: str) -> xr.Dataset:
    """
    Subset the given dataset by a shapefile.

    Parameters:
    - ds: Input xarray Dataset
    - shape_path: Path to the shapefile (.geojson)

    Returns:
    An xarray Dataset subsetted by the shape
    """
    return subset.subset_shape(ds, shape=area_path)
