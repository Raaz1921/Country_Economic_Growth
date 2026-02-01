"""
Inference script for making predictions on new satellite images
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from typing import Dict
import logging

from model import get_model
from dataset import get_data_transforms

logger = logging.getLogger(__name__)


class EconomicPredictor:
    """
    Inference class for economic activity prediction
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        image_size: int = 224
    ):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
            image_size: Input image size
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        self.model = get_model(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        transforms_dict = get_data_transforms(image_size, augment=False)
        self.transform = transforms_dict['test']
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def predict_image(self, image_path: str) -> Dict[str, float]:
        """
        Predict economic indicators for a single image
        
        Args:
            image_path: Path to satellite image
        
        Returns:
            Dictionary of predicted economic indicators
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Convert to dictionary
        results = {
            task_name: float(value.cpu().item())
            for task_name, value in predictions.items()
        }
        
        return results
    
    def predict_batch(self, image_paths: list) -> Dict[str, list]:
        """
        Predict economic indicators for multiple images
        
        Args:
            image_paths: List of paths to satellite images
        
        Returns:
            Dictionary of predictions for each image
        """
        all_predictions = {
            'economic_activity': [],
            'nightlight_intensity': [],
            'building_density': [],
            'road_density': [],
            'vegetation_index': []
        }
        
        for image_path in image_paths:
            predictions = self.predict_image(image_path)
            
            for task_name, value in predictions.items():
                all_predictions[task_name].append(value)
        
        return all_predictions
    
    def predict_and_visualize(self, image_path: str, save_path: str = None):
        """
        Predict and create visualization
        
        Args:
            image_path: Path to satellite image
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get predictions
        predictions = self.predict_image(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        axes[0].imshow(image)
        axes[0].set_title('Satellite Image', fontsize=14)
        axes[0].axis('off')
        
        # Show predictions as bar chart
        tasks = list(predictions.keys())
        values = list(predictions.values())
        
        axes[1].barh(tasks, values, color='steelblue')
        axes[1].set_xlabel('Predicted Value', fontsize=12)
        axes[1].set_title('Economic Activity Predictions', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Format task names
        formatted_tasks = [task.replace('_', ' ').title() for task in tasks]
        axes[1].set_yticks(range(len(tasks)))
        axes[1].set_yticklabels(formatted_tasks)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict economic activity from satellite images')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to satellite image')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                       help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = EconomicPredictor(
        model_path=args.model_path,
        device=args.device
    )
    
    # Make prediction
    if args.visualize:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(args.image_path).stem
        save_path = output_path / f'{image_name}_prediction.png'
        
        predictions = predictor.predict_and_visualize(
            args.image_path,
            save_path=str(save_path)
        )
    else:
        predictions = predictor.predict_image(args.image_path)
    
    # Print predictions
    print("\n" + "=" * 60)
    print("ECONOMIC ACTIVITY PREDICTIONS")
    print("=" * 60)
    
    for task_name, value in predictions.items():
        formatted_name = task_name.replace('_', ' ').title()
        print(f"{formatted_name:30s}: {value:10.4f}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
