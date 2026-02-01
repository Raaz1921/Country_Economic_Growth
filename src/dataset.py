"""
Dataset and DataLoader for Satellite Economic Activity Prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import rasterio
from rasterio.plot import show
import logging

logger = logging.getLogger(__name__)


class SatelliteEconomicDataset(Dataset):
    """
    Dataset for satellite images with economic indicators
    
    Supports multiple formats:
    - GeoTIFF (.tif, .tiff)
    - Regular images (.jpg, .png)
    - Multi-band satellite imagery
    """
    
    def __init__(
        self,
        image_dir: str,
        labels_file: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
        use_rgb_only: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            image_dir: Directory containing satellite images
            labels_file: CSV file with economic indicators
            transform: Optional image transformations
            image_size: Target image size
            use_rgb_only: If True, use only RGB bands; else use all available bands
            normalize: Apply ImageNet normalization
        """
        self.image_dir = Path(image_dir)
        self.labels_df = pd.read_csv(labels_file)
        self.use_rgb_only = use_rgb_only
        
        # Default transform if none provided
        if transform is None:
            transform_list = [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ]
            
            if normalize:
                # ImageNet normalization
                transform_list.append(
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                )
            
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform
        
        # Verify data integrity
        self._verify_data()
        
        logger.info(f"Loaded dataset with {len(self)} samples")
    
    def _verify_data(self):
        """Check that all referenced images exist"""
        missing_images = []
        
        for idx, row in self.labels_df.iterrows():
            image_path = self.image_dir / row['image_filename']
            if not image_path.exists():
                missing_images.append(row['image_filename'])
        
        if missing_images:
            logger.warning(f"Missing {len(missing_images)} images: {missing_images[:5]}...")
            # Remove rows with missing images
            self.labels_df = self.labels_df[
                self.labels_df['image_filename'].apply(
                    lambda x: (self.image_dir / x).exists()
                )
            ].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Get a single sample
        
        Returns:
            image: Preprocessed satellite image tensor
            labels: Dictionary of economic indicators
        """
        row = self.labels_df.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row['image_filename']
        image = self._load_image(image_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Prepare labels
        labels = {
            'economic_activity': float(row.get('economic_activity', 0)),
            'nightlight_intensity': float(row.get('nightlight_intensity', 0)),
            'building_density': float(row.get('building_density', 0)),
            'road_density': float(row.get('road_density', 0)),
            'vegetation_index': float(row.get('vegetation_index', 0))
        }
        
        return image, labels
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load image from various formats"""
        
        # GeoTIFF format
        if image_path.suffix.lower() in ['.tif', '.tiff']:
            return self._load_geotiff(image_path)
        
        # Regular image formats
        else:
            return Image.open(image_path).convert('RGB')
    
    def _load_geotiff(self, image_path: Path) -> Image.Image:
        """Load GeoTIFF and convert to PIL Image"""
        
        with rasterio.open(image_path) as src:
            # Read bands
            if self.use_rgb_only and src.count >= 3:
                # Read RGB bands (typically bands 3, 2, 1 for Landsat/Sentinel)
                # Adjust indices based on your satellite data
                r = src.read(3)  # Red
                g = src.read(2)  # Green
                b = src.read(1)  # Blue
                
                # Stack and normalize
                rgb = np.dstack([r, g, b])
                
            else:
                # Read all bands
                rgb = src.read()
                if rgb.shape[0] > 3:
                    rgb = rgb[:3]  # Take first 3 bands
                rgb = np.transpose(rgb, (1, 2, 0))
            
            # Normalize to 0-255 range
            rgb = self._normalize_array(rgb)
            
            # Convert to PIL Image
            return Image.fromarray(rgb.astype(np.uint8))
    
    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to 0-255 range"""
        arr_min = arr.min()
        arr_max = arr.max()
        
        if arr_max > arr_min:
            arr_normalized = (arr - arr_min) / (arr_max - arr_min) * 255
        else:
            arr_normalized = arr
        
        return arr_normalized


def get_data_transforms(image_size: int = 224, augment: bool = False) -> Dict[str, transforms.Compose]:
    """
    Get data transformation pipelines for train/val/test
    
    Args:
        image_size: Target image size
        augment: Apply data augmentation for training
    
    Returns:
        Dictionary of transforms for each split
    """
    
    # Base transforms
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    # Training transforms with augmentation
    if augment:
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transforms = transforms.Compose(base_transforms)
    
    val_test_transforms = transforms.Compose(base_transforms)
    
    return {
        'train': train_transforms,
        'val': val_test_transforms,
        'test': val_test_transforms
    }


def create_dataloaders(
    data_dir: str,
    labels_file: str,
    batch_size: int = 16,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    image_size: int = 224,
    augment: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders with proper splitting
    
    Returns:
        Dictionary of dataloaders for each split
    """
    
    # Load labels to split data
    labels_df = pd.read_csv(labels_file)
    
    # Shuffle and split
    labels_df = labels_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(labels_df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    train_df = labels_df[:train_end]
    val_df = labels_df[train_end:val_end]
    test_df = labels_df[val_end:]
    
    # Save split CSV files
    data_path = Path(data_dir)
    train_df.to_csv(data_path.parent / 'train_labels.csv', index=False)
    val_df.to_csv(data_path.parent / 'val_labels.csv', index=False)
    test_df.to_csv(data_path.parent / 'test_labels.csv', index=False)
    
    # Get transforms
    transforms_dict = get_data_transforms(image_size, augment)
    
    # Create datasets
    train_dataset = SatelliteEconomicDataset(
        data_dir,
        data_path.parent / 'train_labels.csv',
        transform=transforms_dict['train'],
        image_size=image_size
    )
    
    val_dataset = SatelliteEconomicDataset(
        data_dir,
        data_path.parent / 'val_labels.csv',
        transform=transforms_dict['val'],
        image_size=image_size
    )
    
    test_dataset = SatelliteEconomicDataset(
        data_dir,
        data_path.parent / 'test_labels.csv',
        transform=transforms_dict['test'],
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def collate_fn(batch):
    """Custom collate function for handling dictionary labels"""
    images = torch.stack([item[0] for item in batch])
    
    labels = {}
    for key in batch[0][1].keys():
        labels[key] = torch.tensor([item[1][key] for item in batch], dtype=torch.float32)
    
    return images, labels


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset creation...")
    
    # This is a placeholder - you'll need actual data
    # Example usage when you have data:
    # dataset = SatelliteEconomicDataset(
    #     image_dir='data/raw/images',
    #     labels_file='data/raw/labels.csv'
    # )
    # print(f"Dataset size: {len(dataset)}")
