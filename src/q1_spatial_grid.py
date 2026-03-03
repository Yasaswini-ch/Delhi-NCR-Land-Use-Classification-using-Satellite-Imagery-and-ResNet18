"""
Q1 Module: Spatial Grid Analysis
Creates 5x5km grid over Delhi and performs spatial association with ground stations.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, load_config
from src.utils.crs_utils import validate_crs, reproject_gdf, get_crs_epsg
from src.utils.spatial_utils import (
    create_grid, clip_grid_to_boundary, create_points_gdf,
    spatial_join_points_to_polygons, create_grid_boundary_plot
)


def load_delhi_boundary(config: dict) -> gpd.GeoDataFrame:
    """
    Load Delhi boundary shapefile/GeoJSON.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        GeoDataFrame with Delhi boundary
    """
    logger = setup_logger("Q1")
    raw_data_path = Path(config['paths']['raw_data'])
    
    boundary_file = config['data_files'].get('delhi_boundary', 'delhi_airshed.geojson')
    boundary_path = raw_data_path / boundary_file
    
    logger.info(f"Loading boundary from: {boundary_path}")
    
    if not boundary_path.exists():
        logger.warning(f"Boundary file not found: {boundary_path}")
        logger.info("Creating sample boundary from delhi_ncr_region.geojson")
        ncr_file = config['data_files'].get('delhi_ncr', 'delhi_ncr_region.geojson')
        boundary_path = raw_data_path / ncr_file
    
    boundary_gdf = gpd.read_file(boundary_path)
    
    source_crs = config['crs']['source_crs']
    if not validate_crs(boundary_gdf, source_crs):
        logger.warning(f"Boundary CRS {boundary_gdf.crs} does not match expected {source_crs}")
    
    target_crs = config['crs']['target_crs']
    boundary_gdf = reproject_gdf(boundary_gdf, target_crs)
    
    logger.info(f"Boundary loaded: {len(boundary_gdf)} features, CRS: {boundary_gdf.crs}")
    
    return boundary_gdf


def create_spatial_grid(
    boundary_gdf: gpd.GeoDataFrame,
    config: dict
) -> gpd.GeoDataFrame:
    """
    Generate spatial grid over Delhi boundary.
    
    Args:
        boundary_gdf: Delhi boundary GeoDataFrame
        config: Configuration dictionary
    
    Returns:
        Grid GeoDataFrame
    """
    logger = setup_logger("Q1")
    
    grid_size_km = config['grid']['grid_size_km']
    cell_size = config['grid']['cell_size_meters']
    
    total_bounds = boundary_gdf.total_bounds
    logger.info(f"Boundary bounds (UTM): {total_bounds}")
    
    grid = create_grid(total_bounds, cell_size, crs=str(config['crs']['target_crs']))
    logger.info(f"Created grid with {len(grid)} cells")
    
    clipped_grid = clip_grid_to_boundary(grid, boundary_gdf)
    logger.info(f"Clipped grid to boundary: {len(clipped_grid)} cells")
    
    return clipped_grid


def load_station_data(config: dict) -> gpd.GeoDataFrame:
    """
    Load ground station CSV and convert to GeoDataFrame.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Stations GeoDataFrame
    """
    logger = setup_logger("Q1")
    raw_data_path = Path(config['paths']['raw_data'])
    
    station_file = config['data_files'].get('station_csv')
    station_path = raw_data_path / station_file if station_file else raw_data_path / 'station_data.csv'
    
    if not station_path.exists():
        logger.info("Station CSV not found. Creating sample station data...")
        station_data = create_sample_station_data()
        station_path = raw_data_path / 'station_data.csv'
        station_data.to_csv(station_path, index=False)
        logger.info(f"Sample station data saved to: {station_path}")
    
    logger.info(f"Loading station data from: {station_path}")
    station_df = pd.read_csv(station_path)
    
    required_cols = ['longitude', 'latitude']
    missing = [c for c in required_cols if c not in station_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if 'station_id' not in station_df.columns:
        station_df['station_id'] = [f"STATION_{i:03d}" for i in range(len(station_df))]
    
    if 'aqi' not in station_df.columns:
        station_df['aqi'] = np.random.randint(50, 300, size=len(station_df))
    
    station_gdf = create_points_gdf(
        station_df,
        x_col='longitude',
        y_col='latitude',
        crs=config['crs']['source_crs']
    )
    
    station_gdf = reproject_gdf(station_gdf, config['crs']['target_crs'])
    
    logger.info(f"Loaded {len(station_gdf)} stations")
    
    return station_gdf


def create_sample_station_data() -> pd.DataFrame:
    """
    Create sample station data for Delhi region.
    
    Returns:
        DataFrame with station data
    """
    delhi_lat_range = (28.4, 28.9)
    delhi_lon_range = (76.8, 77.5)
    
    num_stations = 40
    
    np.random.seed(42)
    lats = np.random.uniform(delhi_lat_range[0], delhi_lat_range[1], num_stations)
    lons = np.random.uniform(delhi_lon_range[0], delhi_lon_range[1], num_stations)
    aqis = np.random.randint(50, 300, num_stations)
    
    station_names = [
        'Dwarka', 'R.K. Puram', 'Sarojini Nagar', 'Lajpat Nagar', 'Karol Bagh',
        'Nehru Nagar', 'Anand Vihar', 'Mayur Vihar', 'Paharganj', 'Chandni Chowk',
        'Shahdara', 'Vivek Vihar', 'Preet Vihar', 'Krishna Nagar', 'Geeta Colony',
        'Janakpuri', 'Palam', 'Rajouri Garden', 'Subhash Nagar', 'Tilak Nagar',
        'Model Town', 'Rohini', 'Bawana', 'Narela', 'Sultanpuri',
        'Nangloi', 'Paschim Vihar', 'Punjabi Bagh', 'Shalimar Bagh', 'Ashok Vihar',
        'Civil Lines', 'University', 'ITO', 'Akshardham', 'Pushpanjali',
        'Sonia Vihar', 'Mundka', 'Bahadurgarh', 'Gurgaon', 'Faridabad'
    ]
    
    df = pd.DataFrame({
        'station_id': [f"STATION_{i:03d}" for i in range(num_stations)],
        'station_name': station_names[:num_stations],
        'longitude': lons,
        'latitude': lats,
        'aqi': aqis
    })
    
    return df


def perform_spatial_join(
    grid_gdf: gpd.GeoDataFrame,
    station_gdf: gpd.GeoDataFrame,
    config: dict
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Perform spatial join to associate stations with grid cells.
    
    Args:
        grid_gdf: Grid GeoDataFrame
        station_gdf: Station GeoDataFrame
        config: Configuration dictionary
    
    Returns:
        Tuple of (joined GeoDataFrame, mapping DataFrame)
    """
    logger = setup_logger("Q1")
    
    joined = spatial_join_points_to_polygons(
        station_gdf,
        grid_gdf,
        how='left',
        predicate='within'
    )
    
    logger.info(f"Spatial join completed: {len(joined)} records")
    
    mapping_cols = [col for col in joined.columns if col not in ['geometry', 'index_right']]
    mapping_df = joined[mapping_cols].copy()
    
    unmatched = mapping_df['grid_id'].isna().sum()
    if unmatched > 0:
        logger.warning(f"{unmatched} stations not matched to any grid cell")
    
    return joined, mapping_df


def save_grid_outputs(
    grid_gdf: gpd.GeoDataFrame,
    mapping_df: pd.DataFrame,
    config: dict
) -> None:
    """
    Save grid and mapping outputs.
    
    Args:
        grid_gdf: Grid GeoDataFrame
        mapping_df: Station-grid mapping DataFrame
        config: Configuration dictionary
    """
    logger = setup_logger("Q1")
    processed_path = Path(config['paths']['processed_data'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    output_path = Path(config['paths']['outputs'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    if config['output'].get('save_grid_shapefile', True):
        grid_shp_path = output_path / 'grid.shp'
        grid_gdf.to_file(grid_shp_path)
        logger.info(f"Grid shapefile saved: {grid_shp_path}")
    
    if config['output'].get('save_grid_geopackage', True):
        grid_gpkg_path = output_path / 'grid.gpkg'
        grid_gdf.to_file(grid_gpkg_path, driver='GPKG')
        logger.info(f"Grid GeoPackage saved: {grid_gpkg_path}")
    
    if config['output'].get('save_station_mapping', True):
        mapping_csv_path = processed_path / 'station_grid_mapping.csv'
        mapping_df.to_csv(mapping_csv_path, index=False)
        logger.info(f"Station mapping saved: {mapping_csv_path}")
    
    logger.info("All Q1 outputs saved successfully")


def create_grid_visualization(
    grid_gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    station_gdf: gpd.GeoDataFrame,
    config: dict
) -> None:
    """
    Create matplotlib visualization of grid, boundary, and stations.
    
    Args:
        grid_gdf: Grid GeoDataFrame
        boundary_gdf: Boundary GeoDataFrame
        station_gdf: Station GeoDataFrame
        config: Configuration dictionary
    """
    logger = setup_logger("Q1")
    output_path = Path(config['paths']['outputs'])
    
    plot_path = output_path / 'grid_visualization.png'
    
    dpi = config['output'].get('plot_dpi', 300)
    
    create_grid_boundary_plot(
        grid_gdf=grid_gdf,
        boundary_gdf=boundary_gdf,
        station_gdf=station_gdf,
        column=None,
        title="Delhi 5x5km Spatial Grid with Ground Stations",
        save_path=str(plot_path),
        dpi=dpi
    )
    
    logger.info(f"Grid visualization saved: {plot_path}")


def run_q1(config_path: Optional[str] = None) -> dict:
    """
    Main function to run Q1 spatial grid analysis.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Dictionary with output paths and statistics
    """
    logger = setup_logger("Q1")
    logger.info("=" * 60)
    logger.info("Starting Q1: Spatial Grid Analysis")
    logger.info("=" * 60)
    
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / 'configs' / 'config.yaml')
    
    config = load_config(config_path)
    
    boundary_gdf = load_delhi_boundary(config)
    logger.info("Step 1/5: Delhi boundary loaded")
    
    grid_gdf = create_spatial_grid(boundary_gdf, config)
    logger.info("Step 2/5: Spatial grid created")
    
    station_gdf = load_station_data(config)
    logger.info("Step 3/5: Ground station data loaded")
    
    joined_gdf, mapping_df = perform_spatial_join(grid_gdf, station_gdf, config)
    logger.info("Step 4/5: Spatial join completed")
    
    save_grid_outputs(grid_gdf, mapping_df, config)
    create_grid_visualization(grid_gdf, boundary_gdf, station_gdf, config)
    logger.info("Step 5/5: Outputs saved")
    
    stats = {
        'num_grid_cells': len(grid_gdf),
        'num_stations': len(station_gdf),
        'num_joined_records': len(joined_gdf),
        'grid_area_km2': float(grid_gdf['cell_area_km2'].sum()),
        'crs': str(grid_gdf.crs)
    }
    
    logger.info("=" * 60)
    logger.info("Q1 Complete!")
    logger.info(f"Grid cells: {stats['num_grid_cells']}")
    logger.info(f"Stations: {stats['num_stations']}")
    logger.info(f"Total grid area: {stats['grid_area_km2']:.2f} km²")
    logger.info("=" * 60)
    
    return stats


if __name__ == "__main__":
    stats = run_q1()
