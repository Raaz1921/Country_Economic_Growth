"""
Neural Network Architectures for Economic Activity Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EconomicActivityPredictor(nn.Module):
    """
    Multi-task learning model for predicting economic indicators from satellite imagery
    
    Architecture:
    - Backbone: Pre-trained CNN (ResNet, EfficientNet, etc.)
    - Multi-head output: Separate regression heads for each economic indicator
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        num_indicators: int = 5,
        dropout_rate: float = 0.3,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.num_indicators = num_indicators
        
        # Load pre-trained backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        
        # Get feature dimension from backbone
        self.feature_dim = self._get_feature_dim()
        
        # Shared feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads for each economic indicator
        self.economic_activity_head = self._create_regression_head(hidden_dim, "Economic Activity")
        self.nightlight_head = self._create_regression_head(hidden_dim, "Nightlight Intensity")
        self.building_density_head = self._create_regression_head(hidden_dim, "Building Density")
        self.road_density_head = self._create_regression_head(hidden_dim, "Road Density")
        self.vegetation_head = self._create_regression_head(hidden_dim, "Vegetation Index")
        
        logger.info(f"Initialized {backbone} backbone with {self.num_indicators} output heads")
    
    def _create_backbone(self, backbone: str, pretrained: bool):
        """Create CNN backbone"""
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            # Remove final FC layer
            model = nn.Sequential(*list(model.children())[:-1])
        
        elif backbone == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        
        elif backbone == 'efficientnet_b3':
            model = models.efficientnet_b3(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        
        elif backbone == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            model = model.features
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return model
    
    def _get_feature_dim(self) -> int:
        """Determine feature dimension of backbone"""
        if 'resnet50' in self.backbone_name:
            return 2048
        elif 'resnet101' in self.backbone_name:
            return 2048
        elif 'efficientnet_b3' in self.backbone_name:
            return 1536
        elif 'vgg16' in self.backbone_name:
            return 512 * 7 * 7  # VGG outputs spatial features
        else:
            return 2048  # default
    
    def _create_regression_head(self, input_dim: int, name: str) -> nn.Module:
        """Create a regression head for a specific task"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [batch_size, 3, H, W]
        
        Returns:
            Dictionary of predictions for each economic indicator
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Flatten features
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        
        # Process features through shared layers
        shared_features = self.feature_processor(features)
        
        # Multi-task predictions
        predictions = {
            'economic_activity': self.economic_activity_head(shared_features).squeeze(-1),
            'nightlight_intensity': self.nightlight_head(shared_features).squeeze(-1),
            'building_density': self.building_density_head(shared_features).squeeze(-1),
            'road_density': self.road_density_head(shared_features).squeeze(-1),
            'vegetation_index': self.vegetation_head(shared_features).squeeze(-1)
        }
        
        return predictions


class AttentionEconomicPredictor(nn.Module):
    """
    Advanced model with spatial attention mechanism
    Better for capturing fine-grained economic patterns
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        num_indicators: int = 5,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            # Extract features before global pooling
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            feature_channels = 2048
        else:
            raise NotImplementedError(f"{backbone} not implemented for attention model")
        
        # Spatial attention module
        if use_attention:
            self.attention = SpatialAttention(feature_channels)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Prediction heads
        self.heads = nn.ModuleDict({
            'economic_activity': self._build_head(feature_channels),
            'nightlight_intensity': self._build_head(feature_channels),
            'building_density': self._build_head(feature_channels),
            'road_density': self._build_head(feature_channels),
            'vegetation_index': self._build_head(feature_channels)
        })
    
    def _build_head(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract spatial features
        features = self.backbone(x)  # [B, C, H, W]
        
        # Apply attention
        if self.use_attention:
            features = self.attention(features)
        
        # Global pooling
        pooled = self.global_pool(features).flatten(1)  # [B, C]
        
        # Multi-task predictions
        predictions = {
            name: head(pooled).squeeze(-1)
            for name, head in self.heads.items()
        }
        
        return predictions


class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on economically relevant regions"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate attention map
        attention_map = self.conv(x)  # [B, 1, H, W]
        attention_map = self.sigmoid(attention_map)
        
        # Apply attention
        return x * attention_map


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss function
    Combines losses from different economic indicators
    """
    
    def __init__(self, task_weights: Dict[str, float] = None):
        super().__init__()
        
        self.task_weights = task_weights or {
            'economic_activity': 1.0,
            'nightlight_intensity': 0.5,
            'building_density': 0.5,
            'road_density': 0.3,
            'vegetation_index': 0.3
        }
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        use_mae: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate weighted multi-task loss
        
        Returns:
            Dictionary with total loss and individual task losses
        """
        loss_fn = self.mae_loss if use_mae else self.mse_loss
        
        task_losses = {}
        total_loss = 0
        
        for task_name, weight in self.task_weights.items():
            if task_name in predictions and task_name in targets:
                task_loss = loss_fn(predictions[task_name], targets[task_name])
                task_losses[f'{task_name}_loss'] = task_loss
                total_loss += weight * task_loss
        
        task_losses['total_loss'] = total_loss
        
        return task_losses


def get_model(config) -> nn.Module:
    """Factory function to create model based on configuration"""
    
    if config.get('use_attention', False):
        model = AttentionEconomicPredictor(
            backbone=config.get('backbone', 'resnet50'),
            pretrained=config.get('pretrained', True),
            num_indicators=config.get('num_indicators', 5)
        )
    else:
        model = EconomicActivityPredictor(
            backbone=config.get('backbone', 'resnet50'),
            pretrained=config.get('pretrained', True),
            num_indicators=config.get('num_indicators', 5),
            dropout_rate=config.get('dropout_rate', 0.3),
            hidden_dim=config.get('hidden_dim', 512)
        )
    
    return model


if __name__ == "__main__":
    # Test model
    model = EconomicActivityPredictor()
    x = torch.randn(4, 3, 224, 224)
    outputs = model(x)
    
    print("Model output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
