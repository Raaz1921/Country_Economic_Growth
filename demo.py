"""
Complete End-to-End Demo Pipeline
Demonstrates the full workflow from data generation to prediction
"""

import sys
import logging
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent))

from download_data import SatelliteDataDownloader
from dataset import create_dataloaders
from model import get_model, MultiTaskLoss
from train import Trainer
from evaluate import ModelEvaluator, plot_predictions, save_metrics_report
from predict import EconomicPredictor
from visualize import (
    plot_sample_images,
    plot_data_distribution,
    plot_correlation_matrix,
    plot_model_predictions_comparison
)
from utils import set_seed, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def run_complete_pipeline(quick_demo: bool = True):
    """
    Run the complete end-to-end pipeline
    
    Args:
        quick_demo: If True, use smaller dataset and fewer epochs for quick testing
    """
    
    print("\n" + "="*80)
    print("SATELLITE IMAGE → ECONOMIC ACTIVITY PREDICTOR")
    print("Complete End-to-End Pipeline Demo")
    print("="*80 + "\n")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    if quick_demo:
        num_samples = 50
        num_epochs = 5
        batch_size = 8
        logger.info("Running QUICK DEMO mode (50 samples, 5 epochs)")
    else:
        num_samples = 200
        num_epochs = 30
        batch_size = 16
        logger.info("Running FULL mode (200 samples, 30 epochs)")
    
    config = {
        'backbone': 'resnet50',
        'pretrained': True,
        'num_indicators': 5,
        'dropout_rate': 0.3,
        'hidden_dim': 512,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': 0.0001,
        'weight_decay': 0.00001,
        'optimizer': 'adamw',
        'scheduler': 'plateau',
        'scheduler_patience': 5,
        'early_stopping_patience': 10,
        'use_amp': torch.cuda.is_available(),
        'checkpoint_interval': max(num_epochs // 5, 1),
        'model_dir': 'models',
        'output_dir': 'outputs',
        'data_dir': 'data/raw/images',
        'labels_file': 'data/raw/labels.csv',
        'image_size': 224,
        'num_workers': 2,
        'augmentation': True,
        'train_split': 0.7,
        'val_split': 0.15
    }
    
    # ========================================================================
    # STEP 1: Generate/Download Data
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Generating Sample Data")
    print("-"*80)
    
    downloader = SatelliteDataDownloader(output_dir='data/raw')
    df = downloader.download_sample_data(num_samples=num_samples)
    
    logger.info(f"✓ Generated {num_samples} satellite images with economic indicators")
    
    # ========================================================================
    # STEP 2: Visualize Data
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Visualizing Data Distribution")
    print("-"*80)
    
    try:
        plot_sample_images(
            image_dir=config['data_dir'],
            labels_file=config['labels_file'],
            num_samples=9,
            save_path='outputs/sample_images.png'
        )
        
        plot_data_distribution(
            labels_file=config['labels_file'],
            save_path='outputs/data_distribution.png'
        )
        
        plot_correlation_matrix(
            labels_file=config['labels_file'],
            save_path='outputs/correlation_matrix.png'
        )
        
        logger.info("✓ Created data visualizations")
    except Exception as e:
        logger.warning(f"Visualization failed (non-critical): {e}")
    
    # ========================================================================
    # STEP 3: Create Data Loaders
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Creating Data Loaders")
    print("-"*80)
    
    dataloaders = create_dataloaders(
        data_dir=config['data_dir'],
        labels_file=config['labels_file'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        train_split=config['train_split'],
        val_split=config['val_split'],
        image_size=config['image_size'],
        augment=config['augmentation']
    )
    
    logger.info(f"✓ Created dataloaders:")
    logger.info(f"  - Training samples: {len(dataloaders['train'].dataset)}")
    logger.info(f"  - Validation samples: {len(dataloaders['val'].dataset)}")
    logger.info(f"  - Test samples: {len(dataloaders['test'].dataset)}")
    
    # ========================================================================
    # STEP 4: Create Model
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Initializing Model")
    print("-"*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = get_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✓ Model initialized:")
    logger.info(f"  - Architecture: {config['backbone']}")
    logger.info(f"  - Total parameters: {num_params:,}")
    logger.info(f"  - Trainable parameters: {num_trainable:,}")
    
    # ========================================================================
    # STEP 5: Train Model
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 5: Training Model")
    print("-"*80)
    
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=device
    )
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    trainer.train(num_epochs=num_epochs)
    
    logger.info("✓ Training completed!")
    
    # ========================================================================
    # STEP 6: Evaluate Model
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 6: Evaluating Model on Test Set")
    print("-"*80)
    
    # Load best model
    best_model_path = Path(config['model_dir']) / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best model from checkpoint")
    
    evaluator = ModelEvaluator(model, device=device)
    metrics = evaluator.evaluate(dataloaders['test'])
    
    # Print metrics
    print("\n" + "="*80)
    print("TEST SET RESULTS")
    print("="*80)
    
    for task_name, task_metrics in metrics.items():
        print(f"\n{task_name.upper().replace('_', ' ')}:")
        print("-"*80)
        for metric_name, value in task_metrics.items():
            print(f"  {metric_name:30s}: {value:10.4f}")
    
    print("="*80 + "\n")
    
    # Save metrics
    save_metrics_report(metrics, output_dir='outputs')
    logger.info("✓ Saved evaluation metrics")
    
    # ========================================================================
    # STEP 7: Generate Predictions and Visualizations
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 7: Generating Predictions and Visualizations")
    print("-"*80)
    
    # Get predictions for test set
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
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images = images.to(device)
            predictions = model(images)
            
            for task in all_predictions.keys():
                all_predictions[task].extend(predictions[task].cpu().numpy())
                all_targets[task].extend(labels[task].cpu().numpy())
    
    # Convert to numpy
    import numpy as np
    all_predictions = {k: np.array(v) for k, v in all_predictions.items()}
    all_targets = {k: np.array(v) for k, v in all_targets.items()}
    
    # Create visualizations
    try:
        plot_predictions(
            predictions=all_predictions,
            targets=all_targets,
            output_dir='outputs'
        )
        
        plot_model_predictions_comparison(
            predictions=all_predictions,
            targets=all_targets,
            save_path='outputs/detailed_predictions.png'
        )
        
        logger.info("✓ Created prediction visualizations")
    except Exception as e:
        logger.warning(f"Visualization failed (non-critical): {e}")
    
    # ========================================================================
    # STEP 8: Demo Inference on Single Image
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 8: Running Inference on Sample Image")
    print("-"*80)
    
    # Get first test image
    import pandas as pd
    test_labels_df = pd.read_csv('data/raw/test_labels.csv')
    if len(test_labels_df) > 0:
        sample_image = test_labels_df.iloc[0]['image_filename']
        sample_image_path = Path(config['data_dir']) / sample_image
        
        if sample_image_path.exists():
            predictor = EconomicPredictor(
                model_path=str(best_model_path),
                device=str(device)
            )
            
            predictions = predictor.predict_image(str(sample_image_path))
            
            print("\nSample Prediction:")
            print("-"*80)
            for task, value in predictions.items():
                print(f"  {task.replace('_', ' ').title():30s}: {value:10.4f}")
            print("-"*80)
            
            logger.info("✓ Completed inference demo")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Outputs:")
    print("  📁 models/best_model.pth - Best trained model")
    print("  📊 outputs/prediction_plots.png - Prediction visualizations")
    print("  📊 outputs/training_history.png - Training curves")
    print("  📊 outputs/sample_images.png - Sample satellite images")
    print("  📄 outputs/metrics_report.csv - Evaluation metrics")
    print("\nNext Steps:")
    print("  1. Review the metrics in outputs/metrics_report.csv")
    print("  2. Check visualizations in outputs/")
    print("  3. Use the trained model for predictions on new satellite images")
    print("  4. Fine-tune hyperparameters for better performance")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete satellite prediction pipeline')
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline (default is quick demo)')
    
    args = parser.parse_args()
    
    try:
        run_complete_pipeline(quick_demo=not args.full)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
