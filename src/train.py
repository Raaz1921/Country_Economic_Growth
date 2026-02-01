"""
Training Pipeline for Economic Activity Prediction
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from model import get_model, MultiTaskLoss
from dataset import create_dataloaders, collate_fn
from utils import EarlyStopping, save_checkpoint, load_checkpoint, setup_logging

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training manager for economic activity prediction model
    """
    
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda'
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = MultiTaskLoss(config.get('task_weights'))
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 10),
            verbose=True
        )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'plateau')
        
        if scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.get('scheduler_patience', 5),
                verbose=True
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 50),
                eta_min=1e-6
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': [],
            'economic_activity_loss': [],
            'nightlight_intensity_loss': [],
            'building_density_loss': [],
            'road_density_loss': [],
            'vegetation_index_loss': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    predictions = self.model(images)
                    losses = self.criterion(predictions, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                losses = self.criterion(predictions, labels)
                
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                self.optimizer.step()
            
            # Track losses
            for key, value in losses.items():
                epoch_losses[key].append(value.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': losses['total_loss'].item()})
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        epoch_losses = {
            'total_loss': [],
            'economic_activity_loss': [],
            'nightlight_intensity_loss': [],
            'building_density_loss': [],
            'road_density_loss': [],
            'vegetation_index_loss': []
        }
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                # Forward pass
                predictions = self.model(images)
                losses = self.criterion(predictions, labels)
                
                # Track losses
                for key, value in losses.items():
                    epoch_losses[key].append(value.item())
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train (uses config if not provided)
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 50)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate()
            self.val_losses.append(val_losses)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()
            
            # Log epoch results
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Val Loss: {val_losses['total_loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_model('best_model.pth')
                logger.info(f"Saved best model with val loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Early stopping
            self.early_stopping(val_losses['total_loss'])
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
        
        logger.info("Training completed!")
        self.plot_training_history()
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        save_path = Path(self.config.get('model_dir', 'models')) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def plot_training_history(self):
        """Plot and save training curves"""
        output_dir = Path(self.config.get('output_dir', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot([l['total_loss'] for l in self.train_losses], label='Train')
        axes[0, 0].plot([l['total_loss'] for l in self.val_losses], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Economic activity loss
        axes[0, 1].plot([l['economic_activity_loss'] for l in self.train_losses], label='Train')
        axes[0, 1].plot([l['economic_activity_loss'] for l in self.val_losses], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Economic Activity Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Nightlight loss
        axes[1, 0].plot([l['nightlight_intensity_loss'] for l in self.train_losses], label='Train')
        axes[1, 0].plot([l['nightlight_intensity_loss'] for l in self.val_losses], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Nightlight Intensity Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.learning_rates)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training history plot to {output_dir / 'training_history.png'}")


def main():
    """Main training function"""
    # Setup
    setup_logging()
    
    # Configuration
    config = {
        'backbone': 'resnet50',
        'pretrained': True,
        'num_indicators': 5,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'scheduler': 'plateau',
        'scheduler_patience': 5,
        'early_stopping_patience': 10,
        'use_amp': True,
        'checkpoint_interval': 10,
        'model_dir': 'models',
        'output_dir': 'outputs',
        'data_dir': 'data/raw/images',
        'labels_file': 'data/raw/labels.csv',
        'image_size': 224,
        'num_workers': 4
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=config['data_dir'],
        labels_file=config['labels_file'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config['image_size']
    )
    
    # Create model
    model = get_model(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device
    )
    
    # Train
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
