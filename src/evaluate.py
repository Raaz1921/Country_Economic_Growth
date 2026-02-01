"""
Evaluation and Metrics for Economic Activity Prediction
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation for economic activity prediction models
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(self, dataloader) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on a dataset
        
        Returns:
            Dictionary of metrics for each task
        """
        all_predictions = {
            'economic_activity': [],
            'nightlight_intensity': [],
            'building_density': [],
            'road_density': [],
            'vegetation_index': []
        }
        
        all_targets = {
            'economic_activity': [],
            'nightlight_intensity': [],
            'building_density': [],
            'road_density': [],
            'vegetation_index': []
        }
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)
                
                # Get predictions
                predictions = self.model(images)
                
                # Collect predictions and targets
                for task_name in all_predictions.keys():
                    all_predictions[task_name].extend(
                        predictions[task_name].cpu().numpy()
                    )
                    all_targets[task_name].extend(
                        labels[task_name].cpu().numpy()
                    )
        
        # Calculate metrics for each task
        metrics = {}
        for task_name in all_predictions.keys():
            preds = np.array(all_predictions[task_name])
            targets = np.array(all_targets[task_name])
            
            metrics[task_name] = self._calculate_metrics(preds, targets)
        
        # Overall metrics
        metrics['overall'] = self._calculate_overall_metrics(
            all_predictions, all_targets
        )
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # Correlation coefficients
        pearson_corr, _ = pearsonr(targets, predictions)
        spearman_corr, _ = spearmanr(targets, predictions)
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        mask = targets != 0
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        
        return {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Pearson_Correlation': float(pearson_corr),
            'Spearman_Correlation': float(spearman_corr),
            'MAPE': float(mape)
        }
    
    def _calculate_overall_metrics(
        self,
        predictions_dict: Dict[str, List],
        targets_dict: Dict[str, List]
    ) -> Dict[str, float]:
        """Calculate overall performance across all tasks"""
        
        all_preds = []
        all_targets = []
        
        for task_name in predictions_dict.keys():
            all_preds.extend(predictions_dict[task_name])
            all_targets.extend(targets_dict[task_name])
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        return self._calculate_metrics(all_preds, all_targets)
    
    def predict(self, dataloader) -> Dict[str, np.ndarray]:
        """
        Make predictions on a dataset
        
        Returns:
            Dictionary of predictions for each task
        """
        predictions = {
            'economic_activity': [],
            'nightlight_intensity': [],
            'building_density': [],
            'road_density': [],
            'vegetation_index': []
        }
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Predicting"):
                images = images.to(self.device)
                
                batch_predictions = self.model(images)
                
                for task_name in predictions.keys():
                    predictions[task_name].extend(
                        batch_predictions[task_name].cpu().numpy()
                    )
        
        # Convert to numpy arrays
        predictions = {k: np.array(v) for k, v in predictions.items()}
        
        return predictions


def plot_predictions(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    output_dir: str = 'outputs'
):
    """
    Plot prediction vs actual values for all tasks
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    task_names = list(predictions.keys())
    
    for idx, task_name in enumerate(task_names):
        ax = axes[idx]
        
        preds = predictions[task_name]
        targs = targets[task_name]
        
        # Scatter plot
        ax.scatter(targs, preds, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(targs.min(), preds.min())
        max_val = max(targs.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(targs, preds)
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{task_name.replace("_", " ").title()}\nR² = {r2:.3f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    if len(task_names) < 6:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(output_path / 'prediction_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved prediction plots to {output_path / 'prediction_plots.png'}")


def plot_residuals(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    output_dir: str = 'outputs'
):
    """
    Plot residual distributions for all tasks
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    task_names = list(predictions.keys())
    
    for idx, task_name in enumerate(task_names):
        ax = axes[idx]
        
        preds = predictions[task_name]
        targs = targets[task_name]
        residuals = targs - preds
        
        # Histogram
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        # Calculate statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(
            f'{task_name.replace("_", " ").title()}\n'
            f'Mean: {mean_residual:.3f}, Std: {std_residual:.3f}',
            fontsize=14
        )
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot
    if len(task_names) < 6:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(output_path / 'residual_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved residual plots to {output_path / 'residual_plots.png'}")


def save_metrics_report(
    metrics: Dict[str, Dict[str, float]],
    output_dir: str = 'outputs'
):
    """
    Save metrics to CSV and text report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df_data = []
    for task_name, task_metrics in metrics.items():
        row = {'Task': task_name}
        row.update(task_metrics)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    csv_path = output_path / 'metrics_report.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics CSV to {csv_path}")
    
    # Create text report
    txt_path = output_path / 'metrics_report.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ECONOMIC ACTIVITY PREDICTION - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for task_name, task_metrics in metrics.items():
            f.write(f"\n{task_name.upper().replace('_', ' ')}\n")
            f.write("-" * 80 + "\n")
            
            for metric_name, value in task_metrics.items():
                f.write(f"{metric_name:30s}: {value:10.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    logger.info(f"Saved metrics report to {txt_path}")


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully")
    
    # Test metric calculation
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    
    evaluator = ModelEvaluator(None)  # Dummy
    metrics = evaluator._calculate_metrics(y_pred, y_true)
    
    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
