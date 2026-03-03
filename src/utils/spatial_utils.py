"""
Spatial utilities for grid generation, spatial joins, and raster operations.
"""

from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def create_grid(
    bounds: Tuple[float, float, float, float],
    cell_size: float,
    crs: str = "EPSG:32644"
) -> gpd.GeoDataFrame:
    """
    Create a regular grid within given bounds.
    
    Args:
        bounds: (minx, miny, maxx, maxy) in meters
        cell_size: Cell size in meters
        crs: Coordinate reference system
    
    Returns:
        GeoDataFrame with grid polygons
    """
    minx, miny, maxx, maxy = bounds
    
    x_coords = np.arange(minx, maxx + cell_size, cell_size)
    y_coords = np.arange(miny, maxy + cell_size, cell_size)
    
    polygons = []
    grid_ids = []
    
    for i, x in enumerate(x_coords[:-1]):
        for j, y in enumerate(y_coords[:-1]):
            cell = Polygon([
                (x, y),
                (x + cell_size, y),
                (x + cell_size, y + cell_size),
                (x, y + cell_size),
                (x, y)
            ])
            polygons.append(cell)
            grid_ids.append(f"GRID_{i}_{j}")
    
    gdf = gpd.GeoDataFrame({
        'grid_id': grid_ids,
        'geometry': polygons
    }, crs=crs)
    
    gdf['cell_area_km2'] = gdf.geometry.area / 1_000_000
    gdf['cell_size_m'] = cell_size
    
    return gdf


def clip_grid_to_boundary(
    grid: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Clip grid to boundary polygon.
    
    Args:
        grid: Grid GeoDataFrame
        boundary: Boundary GeoDataFrame
    
    Returns:
        Clipped grid GeoDataFrame
    """
    clipped = gpd.overlay(grid, boundary, how='intersection')
    
    if 'grid_id' not in clipped.columns and 'grid_id_1' in clipped.columns:
        clipped['grid_id'] = clipped['grid_id_1']
    
    return clipped


def create_points_gdf(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    crs: str = "EPSG:4326",
    rename_cols: Optional[Dict[str, str]] = None
) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame from DataFrame with coordinates.
    
    Args:
        df: DataFrame with coordinates
        x_col: Column name for x coordinate (longitude)
        y_col: Column name for y coordinate (latitude)
        crs: CRS for the GeoDataFrame
        rename_cols: Optional dict to rename columns
    
    Returns:
        GeoDataFrame with point geometries
    """
    df_copy = df.copy()
    
    if rename_cols:
        df_copy = df_copy.rename(columns=rename_cols)
    
    geometry = [Point(xy) for xy in zip(df_copy[x_col], df_copy[y_col])]
    gdf = gpd.GeoDataFrame(df_copy, geometry=geometry, crs=crs)
    
    return gdf


def spatial_join_points_to_polygons(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    how: str = "inner",
    op: str = "within",
    predicate: str = "within"
) -> gpd.GeoDataFrame:
    """
    Perform spatial join between points and polygons.
    
    Args:
        points_gdf: Points GeoDataFrame
        polygons_gdf: Polygons GeoDataFrame
        how: Join type ('inner', 'left', 'right', 'outer')
        op: Spatial operation ('within', 'contains', 'intersects')
        predicate: Alias for op (for compatibility)
    
    Returns:
        Joined GeoDataFrame
    """
    joined = gpd.sjoin(
        points_gdf,
        polygons_gdf,
        how=how,
        predicate=predicate
    )
    
    return joined


def extract_raster_values_at_points(
    raster_path: str,
    points_gdf: gpd.GeoDataFrame,
    band_numbers: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Extract raster values at point locations.
    
    Args:
        raster_path: Path to raster file
        points_gdf: Points GeoDataFrame (must be in same CRS as raster)
        band_numbers: List of band numbers to extract
    
    Returns:
        DataFrame with extracted values
    """
    with rasterio.open(raster_path) as src:
        if band_numbers is None:
            band_numbers = list(range(1, src.count + 1))
        
        crs = src.crs.to_epsg() if src.crs else None
        points_transformed = points_gdf
        
        if crs and points_gdf.crs:
            if points_gdf.crs.to_epsg() != crs:
                points_transformed = points_gdf.to_crs(f"EPSG:{crs}")
        
        values_dict = {f'band_{b}': [] for b in band_numbers}
        values_dict['x'] = []
        values_dict['y'] = []
        
        for idx, row in points_transformed.iterrows():
            x, y = row.geometry.x, row.geometry.y
            
            try:
                row_idx, col_idx = src.index(x, y)
                window = Window(col_idx, row_idx, 1, 1)
                
                for band in band_numbers:
                    band_data = src.read(band, window=window)
                    value = band_data[0, 0]
                    nodata = src.nodata
                    
                    if nodata is not None and value == nodata:
                        value = np.nan
                    
                    values_dict[f'band_{band}'].append(value)
            except Exception:
                for band in band_numbers:
                    values_dict[f'band_{band}'].append(np.nan)
            
            values_dict['x'].append(x)
            values_dict['y'].append(y)
    
    return pd.DataFrame(values_dict)


def calculate_raster_statistics_in_polygon(
    raster_path: str,
    polygon: Polygon,
    band_number: int = 1
) -> Dict[str, float]:
    """
    Calculate statistics for raster values within a polygon.
    
    Args:
        raster_path: Path to raster file
        polygon: Shapely polygon
        band_number: Band number to analyze
    
    Returns:
        Dictionary with statistics (mean, std, min, max, median)
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs.to_epsg() if src.crs else None
        
        if crs:
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=f"EPSG:{crs}")
            polygon_transformed = polygon_gdf.to_crs(src.crs).geometry.iloc[0]
        else:
            polygon_transformed = polygon
        
        mask = geometry_mask(
            [polygon_transformed],
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True
        )
        
        band_data = src.read(band_number)
        values = band_data[mask]
        
        nodata = src.nodata
        if nodata is not None:
            values = values[values != nodata]
        
        if len(values) == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'median': np.nan,
                'count': 0
            }
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': int(len(values))
        }


def merge_grid_with_station_data(
    grid_gdf: gpd.GeoDataFrame,
    station_gdf: gpd.GeoDataFrame,
    how: str = "inner"
) -> gpd.GeoDataFrame:
    """
    Merge grid and station data using spatial join.
    
    Args:
        grid_gdf: Grid GeoDataFrame
        station_gdf: Station GeoDataFrame
        how: Join type
    
    Returns:
        Merged GeoDataFrame
    """
    joined = gpd.sjoin(
        grid_gdf,
        station_gdf,
        how=how,
        predicate='within'
    )
    
    return joined


def create_grid_boundary_plot(
    grid_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    station_gdf: Optional[gpd.GeoDataFrame] = None,
    column: Optional[str] = None,
    cmap: str = 'viridis',
    alpha: float = 0.5,
    edgecolor: str = 'black',
    linewidth: float = 0.5,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Delhi Grid Analysis",
    save_path: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    Create visualization of grid with boundary and stations.
    
    Args:
        grid_gdf: Grid GeoDataFrame
        boundary_gdf: Boundary GeoDataFrame
        station_gdf: Station GeoDataFrame (optional)
        column: Column to color by
        cmap: Colormap
        alpha: Transparency
        edgecolor: Edge color
        linewidth: Line width
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    fig, ax = plt.subplots(figsize=figsize)
    
    boundary_gdf.boundary.plot(ax=ax, linewidth=2, color='red', label='Boundary')
    
    if column and column in grid_gdf.columns:
        grid_gdf.plot(
            column=column,
            ax=ax,
            cmap=cmap,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            legend=True
        )
    else:
        grid_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor=edgecolor,
            linewidth=linewidth
        )
    
    if station_gdf is not None and not station_gdf.empty:
        station_gdf.plot(ax=ax, color='blue', markersize=50, marker='o', label='Stations')
    
    ax.set_title(title)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
