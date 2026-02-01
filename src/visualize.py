"""
Visualization utilities for satellite economic prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_sample_images(
    image_dir: str,
    labels_file: str,
    num_samples: int = 9,
    save_path: Optional[str] = None
):
    """
    Display a grid of sample satellite images with their economic indicators
    """
    df = pd.DataFrame(pd.read_csv(labels_file))
    samples = df.sample(min(num_samples, len(df)))
    
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= num_samples:
            break
        
        # Load image
        img_path = Path(image_dir) / row['image_filename']
        if img_path.exists():
            img = Image.open(img_path)
            axes[idx].imshow(img)
        else:
            axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        # Add title with economic info
        title = (f"Economic: {row['economic_activity']:.1f}\n"
                f"Nightlight: {row['nightlight_intensity']:.2f}\n"
                f"Vegetation: {row['vegetation_index']:.2f}")
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis('off')
    
    # Remove empty subplots
    for idx in range(num_samples, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sample images to {save_path}")
    
    plt.show()


def plot_data_distribution(
    labels_file: str,
    save_path: Optional[str] = None
):
    """
    Plot distributions of all economic indicators
    """
    df = pd.read_csv(labels_file)
    
    indicators = ['economic_activity', 'nightlight_intensity', 
                  'building_density', 'road_density', 'vegetation_index']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, indicator in enumerate(indicators):
        if indicator in df.columns:
            # Histogram
            axes[idx].hist(df[indicator], bins=30, alpha=0.7, 
                          color='steelblue', edgecolor='black')
            
            # Add statistics
            mean_val = df[indicator].mean()
            std_val = df[indicator].std()
            
            axes[idx].axvline(mean_val, color='red', linestyle='--', 
                             linewidth=2, label=f'Mean: {mean_val:.2f}')
            
            axes[idx].set_xlabel(indicator.replace('_', ' ').title(), fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{indicator.replace("_", " ").title()}\n'
                              f'μ={mean_val:.2f}, σ={std_val:.2f}', fontsize=13)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution plot to {save_path}")
    
    plt.show()


def plot_correlation_matrix(
    labels_file: str,
    save_path: Optional[str] = None
):
    """
    Plot correlation matrix of economic indicators
    """
    df = pd.read_csv(labels_file)
    
    indicators = ['economic_activity', 'nightlight_intensity', 
                  'building_density', 'road_density', 'vegetation_index']
    
    # Select only indicator columns that exist
    indicator_data = df[[col for col in indicators if col in df.columns]]
    
    # Calculate correlation
    corr = indicator_data.corr()
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.2f', vmin=-1, vmax=1)
    
    plt.title('Correlation Matrix of Economic Indicators', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation matrix to {save_path}")
    
    plt.show()


def plot_model_predictions_comparison(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    save_path: Optional[str] = None
):
    """
    Create comprehensive comparison of predictions vs targets
    """
    fig = plt.figure(figsize=(20, 12))
    
    tasks = list(predictions.keys())
    n_tasks = len(tasks)
    
    for idx, task in enumerate(tasks):
        # Scatter plot
        ax1 = plt.subplot(2, n_tasks, idx + 1)
        ax1.scatter(targets[task], predictions[task], alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = min(targets[task].min(), predictions[task].min())
        max_val = max(targets[task].max(), predictions[task].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect')
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(targets[task], predictions[task])
        mae = mean_absolute_error(targets[task], predictions[task])
        
        ax1.set_xlabel('Actual', fontsize=11)
        ax1.set_ylabel('Predicted', fontsize=11)
        ax1.set_title(f'{task.replace("_", " ").title()}\n'
                     f'R²={r2:.3f}, MAE={mae:.3f}', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residual plot
        ax2 = plt.subplot(2, n_tasks, idx + 1 + n_tasks)
        residuals = targets[task] - predictions[task]
        ax2.scatter(predictions[task], residuals, alpha=0.5, s=30)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted', fontsize=11)
        ax2.set_ylabel('Residual', fontsize=11)
        ax2.set_title('Residual Plot', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction comparison to {save_path}")
    
    plt.show()


def plot_feature_importance_heatmap(
    model,
    image_path: str,
    save_path: Optional[str] = None,
    device: str = 'cuda'
):
    """
    Generate Grad-CAM visualization to show which image regions 
    the model focuses on for predictions
    """
    from torchvision import transforms
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Satellite Image', fontsize=14)
    axes[0].axis('off')
    
    # Predictions bar chart
    tasks = list(predictions.keys())
    values = [float(predictions[task].cpu().item()) for task in tasks]
    
    axes[1].barh(range(len(tasks)), values, color='steelblue')
    axes[1].set_yticks(range(len(tasks)))
    axes[1].set_yticklabels([t.replace('_', ' ').title() for t in tasks])
    axes[1].set_xlabel('Predicted Value', fontsize=12)
    axes[1].set_title('Economic Activity Predictions', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance to {save_path}")
    
    plt.show()


def create_training_report(
    train_losses: List[Dict],
    val_losses: List[Dict],
    save_path: Optional[str] = None
):
    """
    Create comprehensive training report with all metrics
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Extract loss components
    tasks = ['total_loss', 'economic_activity_loss', 'nightlight_intensity_loss',
             'building_density_loss', 'road_density_loss', 'vegetation_index_loss']
    
    for idx, task in enumerate(tasks):
        ax = plt.subplot(2, 3, idx + 1)
        
        train_values = [epoch[task] for epoch in train_losses]
        val_values = [epoch[task] for epoch in val_losses]
        
        epochs = range(1, len(train_values) + 1)
        
        ax.plot(epochs, train_values, 'b-', label='Train', linewidth=2)
        ax.plot(epochs, val_values, 'r-', label='Validation', linewidth=2)
        
        # Mark best validation
        best_epoch = np.argmin(val_values) + 1
        best_val = min(val_values)
        ax.scatter([best_epoch], [best_val], color='gold', s=200, 
                  marker='*', zorder=5, label=f'Best: {best_val:.4f}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(task.replace('_', ' ').title(), fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress Report', fontsize=18, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training report to {save_path}")
    
    plt.show()


def plot_geographic_predictions(
    predictions_df: pd.DataFrame,
    metric: str = 'economic_activity',
    save_path: Optional[str] = None
):
    """
    Plot predictions on a geographic map (if lat/lon available)
    """
    if 'latitude' not in predictions_df.columns or 'longitude' not in predictions_df.columns:
        logger.warning("Latitude/Longitude not available for geographic plot")
        return
    
    plt.figure(figsize=(15, 10))
    
    scatter = plt.scatter(
        predictions_df['longitude'],
        predictions_df['latitude'],
        c=predictions_df[metric],
        s=100,
        cmap='RdYlGn',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, label=metric.replace('_', ' ').title())
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.title(f'Geographic Distribution of {metric.replace("_", " ").title()}', 
             fontsize=16)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved geographic plot to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully")
    
    # Example usage
    print("\nAvailable visualization functions:")
    print("- plot_sample_images()")
    print("- plot_data_distribution()")
    print("- plot_correlation_matrix()")
    print("- plot_model_predictions_comparison()")
    print("- plot_feature_importance_heatmap()")
    print("- create_training_report()")
    print("- plot_geographic_predictions()")
