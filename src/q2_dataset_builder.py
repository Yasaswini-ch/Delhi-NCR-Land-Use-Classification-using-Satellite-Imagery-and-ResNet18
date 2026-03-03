"""
Q2 Module: Dataset Builder
Loads satellite data, ground station AQI, performs time matching,
extracts features, and creates train/test datasets.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, load_config
from src.utils.crs_utils import reproject_gdf
from src.utils.spatial_utils import extract_raster_values_at_points


def load_satellite_images(
    config: dict,
    rgb_dir: str
) -> List[Tuple[np.ndarray, str]]:
    """
    Load satellite RGB images from directory.
    
    Args:
        config: Configuration dictionary
        rgb_dir: Directory containing RGB images
    
    Returns:
        List of (image_array, filename) tuples
    """
    logger = setup_logger("Q2")
    raw_data_path = Path(config['paths']['raw_data'])
    rgb_path = raw_data_path / rgb_dir
    
    if not rgb_path.exists():
        logger.warning(f"RGB directory not found: {rgb_path}")
        return []
    
    image_files = sorted(rgb_path.glob('*.png'))
    logger.info(f"Found {len(image_files)} satellite images")
    
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            images.append((img_array, img_path.name))
        except Exception as e:
            logger.warning(f"Failed to load {img_path.name}: {e}")
    
    return images


def load_satellite_raster(config: dict) -> Optional[rasterio.io.DatasetReader]:
    """
    Load satellite GeoTIFF raster.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Rasterio dataset reader
    """
    logger = setup_logger("Q2")
    raw_data_path = Path(config['paths']['raw_data'])
    
    tiff_file = config['data_files'].get('satellite_tiff', 'worldcover_bbox_delhi_ncr_2021.tif')
    tiff_path = raw_data_path / tiff_file
    
    if not tiff_path.exists():
        logger.warning(f"Satellite raster not found: {tiff_path}")
        return None
    
    dataset = rasterio.open(tiff_path)
    logger.info(f"Loaded raster: {dataset.width}x{dataset.height}, bands: {dataset.count}, CRS: {dataset.crs}")
    
    return dataset


def load_ground_station_aqi(config: dict) -> pd.DataFrame:
    """
    Load ground station AQI data.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        DataFrame with station AQI data
    """
    logger = setup_logger("Q2")
    raw_data_path = Path(config['paths']['raw_data'])
    
    station_file = config['data_files'].get('station_csv', 'station_data.csv')
    station_path = raw_data_path / station_file
    
    if not station_path.exists():
        station_path = raw_data_path / 'station_data.csv'
    
    if not station_path.exists():
        logger.info("Creating sample station AQI data...")
        station_df = create_sample_aqi_data()
    else:
        station_df = pd.read_csv(station_path)
        logger.info(f"Loaded {len(station_df)} station records")
    
    return station_df


def create_sample_aqi_data() -> pd.DataFrame:
    """
    Create sample AQI data for stations.
    
    Returns:
        DataFrame with AQI data
    """
    np.random.seed(42)
    
    num_stations = 40
    num_days = 30
    
    station_ids = [f"STATION_{i:03d}" for i in range(num_stations)]
    
    records = []
    for station_id in station_ids:
        base_aqi = np.random.randint(50, 200)
        
        for day in range(num_days):
            aqi = base_aqi + np.random.randint(-30, 30)
            aqi = max(20, min(500, aqi))
            
            aqi_category = categorize_aqi(aqi)
            
            records.append({
                'station_id': station_id,
                'date': f"2024-01-{day+1:02d}",
                'aqi': aqi,
                'aqi_category': aqi_category,
                'pm25': aqi * np.random.uniform(0.4, 0.6),
                'pm10': aqi * np.random.uniform(0.7, 0.9),
                'no2': aqi * np.random.uniform(0.1, 0.2),
                'o3': aqi * np.random.uniform(0.1, 0.15)
            })
    
    df = pd.DataFrame(records)
    return df


def categorize_aqi(aqi: int) -> str:
    """
    Categorize AQI value.
    
    Args:
        aqi: AQI value
    
    Returns:
        Category string
    """
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'


def load_station_coordinates(config: dict) -> gpd.GeoDataFrame:
    """
    Load station coordinates and create GeoDataFrame.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Station GeoDataFrame
    """
    logger = setup_logger("Q2")
    raw_data_path = Path(config['paths']['raw_data'])
    
    station_file = config['data_files'].get('station_csv', 'station_data.csv')
    station_path = raw_data_path / station_file
    
    if not station_path.exists():
        station_path = raw_data_path / 'station_data.csv'
    
    if station_path.exists():
        df = pd.read_csv(station_path)
    else:
        df = create_sample_aqi_data()[['station_id']].drop_duplicates()
        df['longitude'] = np.random.uniform(76.8, 77.5, len(df))
        df['latitude'] = np.random.uniform(28.4, 28.9, len(df))
    
    station_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs=config['crs']['source_crs']
    )
    
    station_gdf = reproject_gdf(station_gdf, config['crs']['target_crs'])
    
    logger.info(f"Created GeoDataFrame with {len(station_gdf)} stations")
    
    return station_gdf


def extract_image_features(
    images: List[Tuple[np.ndarray, str]],
    station_gdf: gpd.GeoDataFrame,
    image_size: Tuple[int, int] = (64, 64)
) -> pd.DataFrame:
    """
    Extract features from satellite images at station locations.
    
    Args:
        images: List of (image_array, filename) tuples
        station_gdf: Station GeoDataFrame
        image_size: Size to resize images
    
    Returns:
        DataFrame with extracted features
    """
    logger = setup_logger("Q2")
    
    features_list = []
    
    image_positions = parse_image_positions(images)
    
    for idx, station in station_gdf.iterrows():
        x, y = station.geometry.x, station.geometry.y
        
        best_match = find_nearest_image(x, y, image_positions)
        
        if best_match is not None:
            img_array, img_name = best_match
            
            img_resized = np.array(Image.fromarray(img_array).resize(image_size))
            
            r_mean = np.mean(img_resized[:, :, 0])
            g_mean = np.mean(img_resized[:, :, 1])
            b_mean = np.mean(img_resized[:, :, 2])
            
            r_std = np.std(img_resized[:, :, 0])
            g_std = np.std(img_resized[:, :, 1])
            b_std = np.std(img_resized[:, :, 2])
            
            features = {
                'station_id': station.get('station_id', idx),
                'longitude': station.geometry.x,
                'latitude': station.geometry.y,
                'r_mean': r_mean,
                'g_mean': g_mean,
                'b_mean': b_mean,
                'r_std': r_std,
                'g_std': g_std,
                'b_std': b_std,
                'image_name': img_name
            }
            
            for i, band in enumerate(['r', 'g', 'b']):
                features[f'{band}_min'] = np.min(img_resized[:, :, i])
                features[f'{band}_max'] = np.max(img_resized[:, :, i])
            
            features_list.append(features)
    
    df = pd.DataFrame(features_list)
    logger.info(f"Extracted features from {len(df)} stations")
    
    return df


def parse_image_positions(
    images: List[Tuple[np.ndarray, str]]
) -> Dict[Tuple[float, float], Tuple[np.ndarray, str]]:
    """
    Parse image filenames to extract coordinates.
    
    Args:
        images: List of (image_array, filename) tuples
    
    Returns:
        Dictionary mapping (lat, lon) to (image_array, filename)
    """
    positions = {}
    
    for img_array, filename in images:
        try:
            parts = filename.replace('.png', '').split('_')
            lat = float(parts[0])
            lon = float(parts[1])
            positions[(lat, lon)] = (img_array, filename)
        except Exception:
            continue
    
    return positions


def find_nearest_image(
    x: float,
    y: float,
    image_positions: Dict[Tuple[float, float], Tuple[np.ndarray, str]],
    utm_to_geo_factor: float = 111000
) -> Optional[Tuple[np.ndarray, str]]:
    """
    Find nearest satellite image for given coordinates.
    
    Args:
        x: X coordinate (UTM)
        y: Y coordinate (UTM)
        image_positions: Dictionary of image positions
        utm_to_geo_factor: Conversion factor
    
    Returns:
        (image_array, filename) or None
    """
    if not image_positions:
        return None
    
    min_dist = float('inf')
    best_match = None
    
    for (lat, lon), img_data in image_positions.items():
        lat_utm = lat * utm_to_geo_factor
        lon_utm = lon * utm_to_geo_factor
        
        dist = np.sqrt((x - lon_utm)**2 + (y - lat_utm)**2)
        
        if dist < min_dist:
            min_dist = dist
            best_match = img_data
    
    return best_match


def extract_raster_features(
    raster_path: str,
    station_gdf: gpd.GeoDataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Extract raster values at station coordinates.
    
    Args:
        raster_path: Path to raster file
        station_gdf: Station GeoDataFrame
        config: Configuration dictionary
    
    Returns:
        DataFrame with raster values
    """
    logger = setup_logger("Q2")
    
    if not Path(raster_path).exists():
        logger.warning(f"Raster not found: {raster_path}")
        return pd.DataFrame()
    
    station_gdf_4326 = station_gdf.to_crs('EPSG:4326')
    
    features_df = extract_raster_values_at_points(raster_path, station_gdf_4326, band_numbers=[1])
    
    logger.info(f"Extracted raster values for {len(features_df)} stations")
    
    return features_df


def match_time_data(
    aqi_df: pd.DataFrame,
    features_df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Match AQI data with satellite features based on time and station.
    
    Args:
        aqi_df: AQI data DataFrame
        features_df: Satellite features DataFrame
        config: Configuration dictionary
    
    Returns:
        Matched DataFrame
    """
    logger = setup_logger("Q2")
    
    if 'aqi' in aqi_df.columns and 'aqi_category' not in aqi_df.columns:
        aqi_df['aqi_category'] = aqi_df['aqi'].apply(categorize_aqi)
    
    if 'date' in aqi_df.columns:
        matched = pd.merge(aqi_df, features_df, on='station_id', how='inner')
    else:
        aqi_df = aqi_df.drop_duplicates(subset=['station_id'])
        matched = pd.merge(aqi_df, features_df, on='station_id', how='inner')
    
    logger.info(f"Matched {len(matched)} records")
    
    return matched


def prepare_dataset(
    matched_df: pd.DataFrame,
    config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare train and test datasets with proper split.
    
    Args:
        matched_df: Matched DataFrame
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger = setup_logger("Q2")
    
    split_ratio = config['dataset']['train_test_split_ratio']
    random_seed = config['dataset']['random_seed']
    
    label_col = config['dataset']['label_column']
    
    class_counts = matched_df[label_col].value_counts()
    min_class_count = class_counts.min()
    
    if min_class_count >= 2:
        train_df, test_df = train_test_split(
            matched_df,
            train_size=split_ratio,
            random_state=random_seed,
            stratify=matched_df[label_col]
        )
    else:
        logger.warning(f"Cannot stratify - some classes have less than 2 samples. Using random split.")
        train_df, test_df = train_test_split(
            matched_df,
            train_size=split_ratio,
            random_state=random_seed
        )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def visualize_label_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    label_col: str = 'aqi_category'
) -> None:
    """
    Visualize label distribution in train and test sets.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        config: Configuration dictionary
        label_col: Label column name
    """
    logger = setup_logger("Q2")
    output_path = Path(config['paths']['outputs'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    train_counts = train_df[label_col].value_counts()
    test_counts = test_df[label_col].value_counts()
    
    # Determine categories based on label column
    if label_col == 'aqi_category':
        categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
        colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red', 'darkred']
        xlabel = 'AQI Category'
    else:
        # Land-use categories
        categories = ['Built-up', 'Vegetation', 'Water', 'Cropland', 'Others']
        colors = ['gray', 'green', 'blue', 'gold', 'brown']
        xlabel = 'Land-use Category'
    
    train_counts = train_counts.reindex(categories, fill_value=0)
    test_counts = test_counts.reindex(categories, fill_value=0)
    
    axes[0].bar(train_counts.index, train_counts.values, color=colors)
    axes[0].set_title('Training Set Distribution')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(test_counts.index, test_counts.values, color=colors)
    axes[1].set_title('Test Set Distribution')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    save_path = output_path / 'label_distribution.png'
    dpi = config['output'].get('plot_dpi', 300)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Label distribution plot saved: {save_path}")


def save_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict
) -> None:
    """
    Save train and test datasets to CSV files.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        config: Configuration dictionary
    """
    logger = setup_logger("Q2")
    processed_path = Path(config['paths']['processed_data'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    train_path = processed_path / 'train.csv'
    test_path = processed_path / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Train dataset saved: {train_path}")
    logger.info(f"Test dataset saved: {test_path}")


def encode_labels(df: pd.DataFrame, label_col: str = 'aqi_category') -> pd.DataFrame:
    """
    Encode categorical labels to integers.
    
    Args:
        df: DataFrame with labels
        label_col: Label column name
    
    Returns:
        DataFrame with encoded labels
    """
    df = df.copy()
    
    categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    label_mapping = {cat: i for i, cat in enumerate(categories)}
    
    df['label'] = df[label_col].map(label_mapping)
    
    return df


def extract_land_cover_patches_and_labels(
    station_gdf: gpd.GeoDataFrame,
    tiff_path: str,
    patch_size: int = 128
) -> pd.DataFrame:
    """
    Extract 128×128 land-cover patches and assign dominant class labels.
    
    Args:
        station_gdf: Station GeoDataFrame (must be in EPSG:4326)
        tiff_path: Path to land cover TIF file
        patch_size: Size of the square patch (default 128)
    
    Returns:
        DataFrame with extracted patches and dominant class labels
    """
    import rasterio
    from scipy import stats
    
    logger = setup_logger("Q2")
    logger.info(f"Extracting {patch_size}×{patch_size} land-cover patches for {len(station_gdf)} stations")
    
    features = []
    
    with rasterio.open(tiff_path) as src:
        logger.info(f"Raster dimensions: {src.width}×{src.height}, CRS: {src.crs}")
        
        for idx, row in station_gdf.iterrows():
            lon = row.geometry.x
            lat = row.geometry.y
            
            try:
                # Get center pixel coordinates
                row_idx, col_idx = src.index(lon, lat)
                
                # Calculate patch window (64 pixels on each side for 128×128 patch)
                half_size = patch_size // 2
                window = rasterio.windows.Window(
                    col_idx - half_size, 
                    row_idx - half_size, 
                    patch_size, 
                    patch_size
                )
                
                # Check if window is within raster bounds
                if (window.col_off >= 0 and window.row_off >= 0 and 
                    window.col_off + patch_size <= src.width and 
                    window.row_off + patch_size <= src.height):
                    
                    # Read the patch
                    patch_data = src.read(1, window=window)
                    
                    # Calculate dominant class (mode)
                    patch_flat = patch_data.flatten()
                    valid_pixels = patch_flat[patch_flat > 0]  # Exclude nodata values
                    
                    if len(valid_pixels) > 0:
                        dominant_class = stats.mode(valid_pixels, keepdims=False).mode
                        class_counts = np.bincount(valid_pixels)
                        class_proportions = class_counts / len(valid_pixels)
                        
                        # Map ESA class codes to simplified land-use categories
                        simplified_category = map_esa_to_simplified_class(dominant_class)
                        
                        features.append({
                            'station_id': row.get('station_id', idx),
                            'longitude': lon,
                            'latitude': lat,
                            'patch_size': patch_size,
                            'dominant_esa_class': int(dominant_class),
                            'dominant_simplified_class': simplified_category,
                            'class_proportion': float(np.max(class_proportions)),
                            'valid_pixels': len(valid_pixels),
                            'total_pixels': patch_size * patch_size
                        })
                        
                        logger.info(f"Station {idx}: Extracted patch, dominant class = {simplified_category}")
                    else:
                        logger.warning(f"Station {idx}: No valid pixels in patch")
                        
                else:
                    logger.warning(f"Station {idx}: Patch window outside raster bounds")
                    
            except Exception as e:
                logger.error(f"Station {idx}: Failed to extract patch - {e}")
    
    logger.info(f"Successfully extracted patches for {len(features)} stations")
    return pd.DataFrame(features)


def map_esa_to_simplified_class(esa_class: int) -> str:
    """
    Map ESA WorldCover class codes to simplified land-use categories.
    
    Args:
        esa_class: ESA WorldCover class code
    
    Returns:
        Simplified land-use category
    """
    # ESA WorldCover 2021 class mapping
    esa_mapping = {
        10: 'Trees',           # Tree cover
        20: 'Shrubland',       # Shrubland
        30: 'Grassland',       # Grassland
        40: 'Cropland',        # Cropland
        50: 'Built-up',        # Built-up
        60: 'Barren',          # Bare/sparse vegetation
        70: 'Snow',            # Snow and ice
        80: 'Water',           # Permanent water bodies
        90: 'Herbaceous',      # Herbaceous wetland
        95: 'Mangrove',        # Mangroves
        100: 'Moss'            # Moss and lichen
    }
    
    # Simplified mapping
    simplified_mapping = {
        'Trees': 'Vegetation',
        'Shrubland': 'Vegetation', 
        'Grassland': 'Vegetation',
        'Cropland': 'Cropland',
        'Built-up': 'Built-up',
        'Barren': 'Others',
        'Snow': 'Others',
        'Water': 'Water',
        'Herbaceous': 'Vegetation',
        'Mangrove': 'Vegetation',
        'Moss': 'Others'
    }
    
    esa_category = esa_mapping.get(esa_class, 'Others')
    return simplified_mapping.get(esa_category, 'Others')


def extract_tiff_features_at_stations(
    station_gdf: gpd.GeoDataFrame,
    tiff_path: str
) -> pd.DataFrame:
    """
    Extract features from GeoTIFF at station locations.
    
    Args:
        station_gdf: Station GeoDataFrame (must be in EPSG:4326)
        tiff_path: Path to TIF file
    
    Returns:
        DataFrame with extracted features
    """
    import rasterio
    
    features = []
    
    with rasterio.open(tiff_path) as src:
        for idx, row in station_gdf.iterrows():
            lon = row.geometry.x
            lat = row.geometry.y
            
            try:
                row_idx, col_idx = src.index(lon, lat)
                
                window = rasterio.windows.Window(col_idx, row_idx, 1, 1)
                band_data = src.read(1, window=window)
                value = band_data[0, 0]
                
                row_window = rasterio.windows.Window(col_idx - 5, row_idx - 5, 11, 11)
                if row_window.col_off >= 0 and row_window.row_off >= 0:
                    local_data = src.read(1, window=row_window)
                    local_mean = np.mean(local_data)
                    local_std = np.std(local_data)
                    local_min = np.min(local_data)
                    local_max = np.max(local_data)
                else:
                    local_mean = value
                    local_std = 0
                    local_min = value
                    local_max = value
                
                features.append({
                    'station_id': row.get('station_id', idx),
                    'longitude': lon,
                    'latitude': lat,
                    'land_cover_value': value,
                    'local_mean': local_mean,
                    'local_std': local_std,
                    'local_min': local_min,
                    'local_max': local_max
                })
            except Exception as e:
                features.append({
                    'station_id': row.get('station_id', idx),
                    'longitude': lon,
                    'latitude': lat,
                    'land_cover_value': np.nan,
                    'local_mean': np.nan,
                    'local_std': np.nan,
                    'local_min': np.nan,
                    'local_max': np.nan
                })
    
    return pd.DataFrame(features)


def filter_satellite_images_by_region(
    rgb_dir: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Tuple[List[str], int, int]:
    """
    Filter satellite images whose center coordinates fall inside the region.
    
    Args:
        rgb_dir: Directory containing RGB images
        boundary_gdf: Boundary GeoDataFrame (must be in EPSG:4326)
    
    Returns:
        Tuple of (filtered_image_paths, total_images, filtered_images)
    """
    logger = setup_logger("Q2")
    
    rgb_path = Path(rgb_dir)
    all_images = list(rgb_path.glob("*.png"))
    total_images = len(all_images)
    
    logger.info(f"Found {total_images} total satellite images")
    
    filtered_images = []
    
    for img_path in all_images:
        try:
            # Extract lat/lon from filename (e.g., "28.2056_76.8558.png")
            stem = img_path.stem
            lat_str, lon_str = stem.split('_')
            lat = float(lat_str)
            lon = float(lon_str)
            
            # Create point geometry
            point = gpd.points_from_xy([lon], [lat], crs='EPSG:4326')[0]
            point_gdf = gpd.GeoDataFrame(geometry=[point], crs='EPSG:4326')
            
            # Check if point is within boundary
            if boundary_gdf.contains(point).any():
                filtered_images.append(str(img_path))
                
        except Exception as e:
            logger.warning(f"Failed to process {img_path.name}: {e}")
    
    filtered_count = len(filtered_images)
    logger.info(f"Filtered to {filtered_count} images within Delhi-NCR region")
    
    return filtered_images, total_images, filtered_count


def run_q2(config_path: Optional[str] = None) -> dict:
    """
    Main function to run Q2 dataset building with 128×128 patches and dominant class labeling.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Dictionary with statistics
    """
    logger = setup_logger("Q2")
    logger.info("=" * 60)
    logger.info("Starting Q2: Dataset Builder with 128×128 Patches")
    logger.info("=" * 60)
    
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / 'configs' / 'config.yaml')
    
    config = load_config(config_path)
    
    logger.info("Step 1/7: Loading station data and Delhi-NCR boundary...")
    aqi_df = load_ground_station_aqi(config)
    station_gdf = load_station_coordinates(config)
    
    # Load Delhi-NCR boundary for image filtering
    from q1_spatial_grid import load_boundary
    boundary_gdf = load_boundary(config)
    boundary_gdf_4326 = boundary_gdf.to_crs('EPSG:4326')
    
    station_gdf_4326 = station_gdf.to_crs('EPSG:4326')
    
    logger.info("Step 2/7: Filtering satellite images by region...")
    rgb_dir = config['data_files'].get('rgb_images', 'rgb')
    raw_data_path = Path(config['paths']['raw_data'])
    rgb_full_path = raw_data_path / rgb_dir
    
    filtered_images, total_images, filtered_count = filter_satellite_images_by_region(
        str(rgb_full_path), boundary_gdf_4326
    )
    
    logger.info(f"Image filtering: {total_images} -> {filtered_count} images")
    
    logger.info("Step 3/7: Extracting 128×128 land-cover patches and labels...")
    tiff_file = config['data_files'].get('satellite_tiff', 'worldcover_bbox_delhi_ncr_2021.tif')
    tiff_path = raw_data_path / tiff_file
    
    # Use new patch extraction function
    land_cover_df = extract_land_cover_patches_and_labels(station_gdf_4326, str(tiff_path))
    logger.info(f"Extracted land-cover patches for {len(land_cover_df)} stations")
    
    logger.info("Step 4/7: Loading filtered RGB satellite images...")
    if filtered_images:
        # Load only filtered images
        images = []
        for img_path in filtered_images:
            try:
                img = Image.open(img_path)
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
        
        if images:
            img_features_df = extract_image_features(images, station_gdf)
            # Merge with land cover data
            land_cover_df = pd.merge(land_cover_df, img_features_df, 
                                   on=['station_id', 'longitude', 'latitude'], how='left')
            logger.info("Added RGB image features")
    else:
        logger.warning("No filtered RGB images available")
    
    logger.info("Step 5/7: Matching AQI with land-cover features...")
    matched_df = pd.merge(aqi_df, land_cover_df, on='station_id', how='inner')
    
    # Use simplified land-use class as labels instead of AQI
    matched_df['land_use_category'] = matched_df['dominant_simplified_class']
    
    # Encode land-use labels
    land_use_categories = ['Built-up', 'Vegetation', 'Water', 'Cropland', 'Others']
    label_mapping = {cat: i for i, cat in enumerate(land_use_categories)}
    matched_df['label'] = matched_df['land_use_category'].map(label_mapping)
    
    logger.info(f"Matched {len(matched_df)} records")
    
    logger.info("Step 6/7: Creating 60/40 train-test split...")
    # Update config for 60/40 split
    config['dataset']['train_test_split_ratio'] = 0.6
    train_df, test_df = prepare_dataset(matched_df, config)
    
    logger.info("Step 7/7: Saving datasets and visualizing class distribution...")
    save_datasets(train_df, test_df, config)
    visualize_label_distribution(train_df, test_df, config, label_col='land_use_category')
    
    stats = {
        'total_images_before_filter': total_images,
        'total_images_after_filter': filtered_count,
        'total_samples': len(matched_df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'num_features': len([c for c in matched_df.columns if c not in ['station_id', 'date', 'aqi', 'aqi_category', 'pm25', 'pm10', 'no2', 'o3', 'label', 'longitude', 'latitude', 'image_name', 'land_use_category', 'dominant_simplified_class']]),
        'num_classes': len(land_use_categories),
        'class_distribution': matched_df['land_use_category'].value_counts().to_dict()
    }
    
    logger.info("=" * 60)
    logger.info("Q2 Complete!")
    logger.info(f"Image filtering: {stats['total_images_before_filter']} -> {stats['total_images_after_filter']}")
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Train/Test split: {stats['train_samples']}/{stats['test_samples']}")
    logger.info(f"Features: {stats['num_features']}")
    logger.info(f"Classes: {stats['num_classes']}")
    logger.info(f"Class distribution: {stats['class_distribution']}")
    logger.info("=" * 60)
    
    return stats


def create_synthetic_features(station_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Create synthetic features when satellite data is unavailable.
    
    Args:
        station_gdf: Station GeoDataFrame
    
    Returns:
        DataFrame with synthetic features
    """
    np.random.seed(42)
    
    features = []
    for idx, station in station_gdf.iterrows():
        features.append({
            'station_id': station.get('station_id', f"STATION_{idx}"),
            'longitude': station.geometry.x,
            'latitude': station.geometry.y,
            'r_mean': np.random.uniform(100, 200),
            'g_mean': np.random.uniform(80, 150),
            'b_mean': np.random.uniform(60, 120),
            'r_std': np.random.uniform(20, 50),
            'g_std': np.random.uniform(15, 40),
            'b_std': np.random.uniform(10, 30),
            'r_min': np.random.uniform(50, 100),
            'r_max': np.random.uniform(180, 250),
            'g_min': np.random.uniform(40, 80),
            'g_max': np.random.uniform(150, 220),
            'b_min': np.random.uniform(30, 60),
            'b_max': np.random.uniform(100, 180)
        })
    
    return pd.DataFrame(features)


if __name__ == "__main__":
    stats = run_q2()
