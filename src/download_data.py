"""
Data Download and Preparation Script
Downloads satellite imagery and prepares economic indicators
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging
from typing import Optional, List
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import json

logger = logging.getLogger(__name__)


class SatelliteDataDownloader:
    """
    Download satellite data from various sources
    
    Supported sources:
    - Landsat 8/9 (via USGS Earth Explorer API)
    - Sentinel-2 (via Copernicus Open Access Hub)
    - Planet Labs (requires API key)
    - NASA VIIRS Nightlights
    """
    
    def __init__(self, output_dir: str = 'data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized data downloader. Output: {self.output_dir}")
    
    def download_sample_data(self, num_samples: int = 100):
        """
        Generate synthetic sample data for testing and demonstration
        
        This creates realistic-looking satellite images and economic indicators
        that can be used to test the model pipeline.
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        # Create sample images and labels
        labels_data = []
        
        for i in range(num_samples):
            # Generate synthetic satellite image
            image = self._generate_synthetic_satellite_image()
            
            # Save image
            image_filename = f'sample_{i:04d}.png'
            image_path = self.images_dir / image_filename
            Image.fromarray(image).save(image_path)
            
            # Generate correlated economic indicators
            # These are synthetic but roughly realistic relationships
            base_activity = np.random.uniform(20, 95)
            
            labels_data.append({
                'image_filename': image_filename,
                'economic_activity': base_activity + np.random.normal(0, 5),
                'nightlight_intensity': (base_activity / 100) ** 0.7 + np.random.normal(0, 0.1),
                'building_density': (base_activity / 100) ** 0.5 + np.random.normal(0, 0.08),
                'road_density': base_activity / 5 + np.random.normal(0, 2),
                'vegetation_index': 0.8 - (base_activity / 120) + np.random.normal(0, 0.15),
                'latitude': np.random.uniform(-60, 60),
                'longitude': np.random.uniform(-180, 180),
                'date': (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d'),
                'region': f'Region_{i % 10}'
            })
        
        # Create labels DataFrame
        df = pd.DataFrame(labels_data)
        
        # Clip values to realistic ranges
        df['nightlight_intensity'] = df['nightlight_intensity'].clip(0, 1)
        df['building_density'] = df['building_density'].clip(0, 1)
        df['vegetation_index'] = df['vegetation_index'].clip(-1, 1)
        df['economic_activity'] = df['economic_activity'].clip(0, 100)
        df['road_density'] = df['road_density'].clip(0, 30)
        
        # Save labels
        labels_path = self.output_dir / 'labels.csv'
        df.to_csv(labels_path, index=False)
        
        logger.info(f"Generated {num_samples} samples")
        logger.info(f"Images saved to: {self.images_dir}")
        logger.info(f"Labels saved to: {labels_path}")
        
        # Print statistics
        print("\n" + "=" * 60)
        print("SAMPLE DATA STATISTICS")
        print("=" * 60)
        print(df.describe())
        print("=" * 60 + "\n")
        
        return df
    
    def _generate_synthetic_satellite_image(self, size: tuple = (224, 224)) -> np.ndarray:
        """
        Generate a synthetic satellite image that looks realistic
        
        Simulates typical satellite image characteristics:
        - Land use patterns (urban, vegetation, water)
        - Spatial coherence
        - Realistic color distribution
        """
        # Create base layers
        np.random.seed(np.random.randint(0, 10000))
        
        # Urban development pattern (clustered)
        urban = self._generate_clustered_pattern(size, num_clusters=3, cluster_size=40)
        
        # Vegetation pattern
        vegetation = self._generate_smooth_noise(size)
        
        # Water bodies (less common)
        water = (self._generate_clustered_pattern(size, num_clusters=1, cluster_size=30) > 0.7).astype(float)
        
        # Combine into RGB
        # Red channel: urban + soil
        red = (urban * 180 + vegetation * 100 + water * 30).clip(0, 255).astype(np.uint8)
        
        # Green channel: vegetation + urban
        green = (vegetation * 200 + urban * 120 + water * 60).clip(0, 255).astype(np.uint8)
        
        # Blue channel: water + urban
        blue = (water * 200 + urban * 100 + vegetation * 80).clip(0, 255).astype(np.uint8)
        
        # Stack RGB
        image = np.stack([red, green, blue], axis=-1)
        
        # Add realistic noise
        noise = np.random.randint(-10, 10, size=image.shape)
        image = (image + noise).clip(0, 255).astype(np.uint8)
        
        return image
    
    def _generate_clustered_pattern(
        self,
        size: tuple,
        num_clusters: int = 5,
        cluster_size: int = 30
    ) -> np.ndarray:
        """Generate spatially clustered pattern (for urban areas)"""
        pattern = np.zeros(size)
        
        for _ in range(num_clusters):
            # Random cluster center
            cy, cx = np.random.randint(0, size[0]), np.random.randint(0, size[1])
            
            # Create cluster
            y, x = np.ogrid[:size[0], :size[1]]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            cluster = np.exp(-(dist**2) / (2 * cluster_size**2))
            
            pattern += cluster
        
        return pattern / pattern.max() if pattern.max() > 0 else pattern
    
    def _generate_smooth_noise(self, size: tuple) -> np.ndarray:
        """Generate smooth Perlin-like noise"""
        from scipy.ndimage import gaussian_filter
        
        noise = np.random.rand(*size)
        smooth_noise = gaussian_filter(noise, sigma=15)
        
        return (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
    
    def download_nightlights_data(
        self,
        region_bounds: tuple,
        output_filename: str = 'nightlights.tif'
    ):
        """
        Download VIIRS nighttime lights data
        
        Note: This is a placeholder. Actual implementation would use:
        - NASA VIIRS API
        - NOAA data portal
        - Google Earth Engine
        """
        logger.warning("Nightlights download not implemented. Use generate_sample_data() for testing.")
        pass
    
    def download_landsat_data(
        self,
        region_bounds: tuple,
        start_date: str,
        end_date: str
    ):
        """
        Download Landsat imagery
        
        Note: This requires USGS Earth Explorer credentials
        """
        logger.warning("Landsat download not implemented. Use generate_sample_data() for testing.")
        pass


class EconomicDataCollector:
    """
    Collect economic data from various sources
    
    Sources:
    - World Bank API
    - OpenStreetMap for infrastructure
    - Census data
    - VIIRS nightlights
    """
    
    def __init__(self):
        self.data = {}
    
    def fetch_world_bank_data(
        self,
        country_code: str,
        indicators: List[str] = ['NY.GDP.MKTP.CD']
    ):
        """
        Fetch economic data from World Bank API
        
        Example indicators:
        - NY.GDP.MKTP.CD: GDP
        - SL.UEM.TOTL.ZS: Unemployment rate
        - SP.URB.TOTL.IN.ZS: Urban population
        """
        base_url = "http://api.worldbank.org/v2/country"
        
        for indicator in indicators:
            url = f"{base_url}/{country_code}/indicator/{indicator}?format=json"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self.data[indicator] = data
                    logger.info(f"Fetched {indicator} for {country_code}")
                else:
                    logger.warning(f"Failed to fetch {indicator}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")
        
        return self.data
    
    def collect_osm_infrastructure(self, bbox: tuple):
        """
        Collect infrastructure data from OpenStreetMap
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        """
        logger.warning("OSM data collection not implemented. Use synthetic data for testing.")
        pass


def main():
    parser = argparse.ArgumentParser(description='Download satellite and economic data')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                       help='Output directory for data')
    parser.add_argument('--mode', type=str, default='synthetic',
                       choices=['synthetic', 'landsat', 'sentinel'],
                       help='Data source mode')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize downloader
    downloader = SatelliteDataDownloader(output_dir=args.output_dir)
    
    # Download data based on mode
    if args.mode == 'synthetic':
        downloader.download_sample_data(num_samples=args.num_samples)
        logger.info("✓ Synthetic data generated successfully!")
        logger.info("  You can now train the model using this data.")
    
    elif args.mode == 'landsat':
        logger.info("Landsat download not yet implemented.")
        logger.info("Please use --mode synthetic for testing.")
    
    elif args.mode == 'sentinel':
        logger.info("Sentinel download not yet implemented.")
        logger.info("Please use --mode synthetic for testing.")


if __name__ == "__main__":
    main()
