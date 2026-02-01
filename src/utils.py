"""
Utility Functions for Satellite Economic Prediction
"""

import torch
import numpy as np
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format
        )


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss: float):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    scheduler=None
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model,
    optimizer=None,
    scheduler=None,
    device: str = 'cuda'
):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def save_config(config: Dict, filepath: str):
    """Save configuration to file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix in ['.yaml', '.yml']:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {filepath.suffix}")


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MetricsTracker:
    """Track training metrics across epochs"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_latest(self, key: str):
        """Get latest value for a metric"""
        return self.metrics[key][-1] if key in self.metrics else None
    
    def get_all(self, key: str):
        """Get all values for a metric"""
        return self.metrics.get(key, [])
    
    def save(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_directory_structure(base_dir: str):
    """Create project directory structure"""
    base_path = Path(base_dir)
    
    directories = [
        'data/raw/images',
        'data/raw/labels',
        'data/processed',
        'models',
        'outputs/plots',
        'outputs/predictions',
        'logs',
        'configs'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    set_seed(42)
    print("✓ Random seed set")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3)
    for i in range(5):
        val_loss = 0.5 + i * 0.1
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"✓ Early stopping triggered at iteration {i}")
            break
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update(loss=0.5, accuracy=0.8)
    tracker.update(loss=0.4, accuracy=0.85)
    print(f"✓ Metrics tracker: {tracker.metrics}")
    
    print("All tests passed!")
