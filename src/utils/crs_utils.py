"""
Coordinate Reference System (CRS) utilities for spatial operations.
Handles CRS validation, transformation, and reprojection.
"""

from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from rasterio.crs import CRS as RasterioCRS
from shapely.geometry import Point, Polygon
from shapely.ops import transform


def validate_crs(gdf: gpd.GeoDataFrame, expected_epsg: Optional[str] = None) -> bool:
    """
    Validate CRS of a GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame to validate
        expected_epsg: Expected EPSG code (e.g., 'EPSG:4326')
    
    Returns:
        True if CRS is valid
    """
    if gdf.crs is None:
        return False
    
    if expected_epsg:
        return str(gdf.crs).upper() == expected_epsg.upper()
    
    return True


def get_crs_epsg(crs_string: str) -> str:
    """
    Extract EPSG code from CRS string.
    
    Args:
        crs_string: CRS string (e.g., 'EPSG:4326' or '4326')
    
    Returns:
        Standardized EPSG string
    """
    if 'EPSG:' in crs_string.upper():
        return f"EPSG:{crs_string.split(':')[-1]}"
    return f"EPSG:{crs_string}"


def reproject_gdf(
    gdf: gpd.GeoDataFrame,
    target_crs: Union[str, int],
    source_crs: Optional[Union[str, int]] = None
) -> gpd.GeoDataFrame:
    """
    Reproject GeoDataFrame to target CRS.
    
    Args:
        gdf: GeoDataFrame to reproject
        target_crs: Target CRS (EPSG code or proj4 string)
        source_crs: Source CRS if gdf has no CRS set
    
    Returns:
        Reprojected GeoDataFrame
    """
    if gdf.crs is None and source_crs is None:
        raise ValueError("Source CRS not provided and GeoDataFrame has no CRS")
    
    if source_crs:
        gdf = gdf.set_crs(source_crs, inplace=False)
    
    return gdf.to_crs(target_crs)


def create_utm_crs(longitude: float, latitude: float) -> str:
    """
    Create UTM CRS for given longitude and latitude.
    
    Args:
        longitude: Longitude value
        latitude: Latitude value
    
    Returns:
        EPSG code for appropriate UTM zone
    """
    if longitude < 0:
        zone_number = int((longitude + 180) / 6) + 1
        if latitude > 0:
            return f"EPSG:{32600 + zone_number}"
        else:
            return f"EPSG:{32700 + zone_number}"
    else:
        zone_number = int(longitude / 6) + 1
        if latitude > 0:
            return f"EPSG:{32600 + zone_number}"
        else:
            return f"EPSG:{32700 + zone_number}"


def get_raster_crs(raster_path: str) -> str:
    """
    Get CRS from a raster file.
    
    Args:
        raster_path: Path to raster file
    
    Returns:
        CRS as EPSG string
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
    return crs.to_epsg() if crs else None


def transform_crs(
    geometry: Union[Point, Polygon],
    source_crs: Union[str, int],
    target_crs: Union[str, int]
) -> Union[Point, Polygon]:
    """
    Transform geometry from source CRS to target CRS.
    
    Args:
        geometry: Shapely geometry
        source_crs: Source CRS
        target_crs: Target CRS
    
    Returns:
        Transformed geometry
    """
    project = pyproj.Transformer.from_crs(
        pyproj.CRS(source_crs),
        pyproj.CRS(target_crs),
        always_xy=True
    ).transform
    
    return transform(project, geometry)


def get_bounds_crs(
    bounds: Tuple[float, float, float, float],
    source_crs: str,
    target_crs: str
) -> Tuple[float, float, float, float]:
    """
    Transform bounds from source CRS to target CRS.
    
    Args:
        bounds: (minx, miny, maxx, maxy)
        source_crs: Source CRS
        target_crs: Target CRS
    
    Returns:
        Transformed bounds
    """
    minx, miny, maxx, maxy = bounds
    
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS(source_crs),
        pyproj.CRS(target_crs),
        always_xy=True
    )
    
    new_minx, new_miny = transformer.transform(minx, miny)
    new_maxx, new_maxy = transformer.transform(maxx, maxy)
    
    return (new_minx, new_miny, new_maxx, new_maxy)


def calculate_area_sqkm(geometry: Polygon, crs_epsg: int = 32644) -> float:
    """
    Calculate area of polygon in square kilometers.
    
    Args:
        geometry: Shapely Polygon
        crs_epsg: CRS EPSG code (must be projected for area calculation)
    
    Returns:
        Area in square kilometers
    """
    if not isinstance(geometry, Polygon):
        return 0.0
    
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=f"EPSG:{crs_epsg}")
    area_sqm = gdf.geometry.area.iloc[0]
    return area_sqm / 1_000_000


def get_centroid_coordinates(
    geometry: Union[Point, Polygon],
    crs: str = "EPSG:4326"
) -> Tuple[float, float]:
    """
    Get centroid coordinates of geometry.
    
    Args:
        geometry: Shapely geometry
        crs: CRS of the geometry
    
    Returns:
        (longitude, latitude) tuple
    """
    if isinstance(geometry, Point):
        return (geometry.x, geometry.y)
    elif isinstance(geometry, Polygon):
        centroid = geometry.centroid
        return (centroid.x, centroid.y)
    else:
        raise ValueError(f"Unsupported geometry type: {type(geometry)}")
