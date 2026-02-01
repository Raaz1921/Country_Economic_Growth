"""
Test Suite for Satellite Economic Predictor
Run tests to verify all components are working correctly
"""

import sys
import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.append('src')

from model import EconomicActivityPredictor, MultiTaskLoss, get_model
from dataset import SatelliteEconomicDataset, create_dataloaders
from download_data import SatelliteDataDownloader
from utils import set_seed, EarlyStopping
from evaluate import ModelEvaluator


class TestDataGeneration(unittest.TestCase):
    """Test data generation and loading"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        set_seed(42)
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        downloader = SatelliteDataDownloader(output_dir=self.temp_dir)
        df = downloader.download_sample_data(num_samples=10)
        
        # Check DataFrame
        self.assertEqual(len(df), 10)
        self.assertIn('image_filename', df.columns)
        self.assertIn('economic_activity', df.columns)
        
        # Check images were created
        images_dir = Path(self.temp_dir) / 'images'
        self.assertTrue(images_dir.exists())
        self.assertEqual(len(list(images_dir.glob('*.png'))), 10)
    
    def test_data_value_ranges(self):
        """Test that generated data is within expected ranges"""
        downloader = SatelliteDataDownloader(output_dir=self.temp_dir)
        df = downloader.download_sample_data(num_samples=20)
        
        # Check economic activity
        self.assertTrue((df['economic_activity'] >= 0).all())
        self.assertTrue((df['economic_activity'] <= 100).all())
        
        # Check nightlight intensity
        self.assertTrue((df['nightlight_intensity'] >= 0).all())
        self.assertTrue((df['nightlight_intensity'] <= 1).all())
        
        # Check vegetation index
        self.assertTrue((df['vegetation_index'] >= -1).all())
        self.assertTrue((df['vegetation_index'] <= 1).all())


class TestModel(unittest.TestCase):
    """Test model architecture"""
    
    def setUp(self):
        set_seed(42)
        self.config = {
            'backbone': 'resnet50',
            'pretrained': False,  # Faster for testing
            'num_indicators': 5,
            'dropout_rate': 0.3,
            'hidden_dim': 512
        }
    
    def test_model_creation(self):
        """Test model can be created"""
        model = get_model(self.config)
        self.assertIsInstance(model, EconomicActivityPredictor)
    
    def test_forward_pass(self):
        """Test forward pass with dummy data"""
        model = get_model(self.config)
        model.eval()
        
        # Create dummy input
        x = torch.randn(4, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(x)
        
        # Check outputs
        self.assertIn('economic_activity', outputs)
        self.assertEqual(outputs['economic_activity'].shape, (4,))
        
        # Check all tasks
        expected_tasks = ['economic_activity', 'nightlight_intensity', 
                         'building_density', 'road_density', 'vegetation_index']
        for task in expected_tasks:
            self.assertIn(task, outputs)
            self.assertEqual(outputs[task].shape, (4,))
    
    def test_model_parameters(self):
        """Test model has trainable parameters"""
        model = get_model(self.config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() 
                              if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        print(f"Model parameters: {total_params:,}")


class TestLoss(unittest.TestCase):
    """Test loss function"""
    
    def test_multitask_loss(self):
        """Test multi-task loss calculation"""
        loss_fn = MultiTaskLoss()
        
        # Create dummy predictions and targets
        predictions = {
            'economic_activity': torch.randn(4),
            'nightlight_intensity': torch.randn(4),
            'building_density': torch.randn(4),
            'road_density': torch.randn(4),
            'vegetation_index': torch.randn(4)
        }
        
        targets = {
            'economic_activity': torch.randn(4),
            'nightlight_intensity': torch.randn(4),
            'building_density': torch.randn(4),
            'road_density': torch.randn(4),
            'vegetation_index': torch.randn(4)
        }
        
        # Calculate loss
        losses = loss_fn(predictions, targets)
        
        # Check losses
        self.assertIn('total_loss', losses)
        self.assertIsInstance(losses['total_loss'], torch.Tensor)
        self.assertGreater(losses['total_loss'].item(), 0)


class TestDataset(unittest.TestCase):
    """Test dataset class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate test data
        downloader = SatelliteDataDownloader(output_dir=self.temp_dir)
        self.df = downloader.download_sample_data(num_samples=10)
        
        self.image_dir = Path(self.temp_dir) / 'images'
        self.labels_file = Path(self.temp_dir) / 'labels.csv'
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_creation(self):
        """Test dataset can be created"""
        dataset = SatelliteEconomicDataset(
            image_dir=str(self.image_dir),
            labels_file=str(self.labels_file)
        )
        
        self.assertEqual(len(dataset), 10)
    
    def test_dataset_getitem(self):
        """Test getting items from dataset"""
        dataset = SatelliteEconomicDataset(
            image_dir=str(self.image_dir),
            labels_file=str(self.labels_file)
        )
        
        # Get first item
        image, labels = dataset[0]
        
        # Check image
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        
        # Check labels
        self.assertIsInstance(labels, dict)
        self.assertIn('economic_activity', labels)


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_early_stopping(self):
        """Test early stopping mechanism"""
        early_stopping = EarlyStopping(patience=3)
        
        # Simulate training with improving loss
        early_stopping(0.5)
        self.assertFalse(early_stopping.early_stop)
        
        early_stopping(0.4)
        self.assertFalse(early_stopping.early_stop)
        
        # Simulate plateau
        early_stopping(0.41)
        early_stopping(0.42)
        early_stopping(0.43)
        early_stopping(0.44)
        
        self.assertTrue(early_stopping.early_stop)
    
    def test_seed_setting(self):
        """Test that seed setting works"""
        set_seed(42)
        r1 = torch.rand(5)
        
        set_seed(42)
        r2 = torch.rand(5)
        
        self.assertTrue(torch.allclose(r1, r2))


class TestEvaluation(unittest.TestCase):
    """Test evaluation metrics"""
    
    def test_metrics_calculation(self):
        """Test metrics are calculated correctly"""
        # Create dummy evaluator
        evaluator = ModelEvaluator(None)
        
        # Test data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
        
        metrics = evaluator._calculate_metrics(y_pred, y_true)
        
        # Check metrics exist
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        
        # Check reasonable values
        self.assertGreater(metrics['R2'], 0.9)  # Should be high for this data
        self.assertLess(metrics['MAE'], 0.2)  # Should be low


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        set_seed(42)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to prediction"""
        print("\n" + "="*60)
        print("Running End-to-End Integration Test")
        print("="*60)
        
        # 1. Generate data
        print("1. Generating data...")
        downloader = SatelliteDataDownloader(output_dir=self.temp_dir)
        df = downloader.download_sample_data(num_samples=20)
        
        # 2. Create model
        print("2. Creating model...")
        config = {
            'backbone': 'resnet50',
            'pretrained': False,
            'num_indicators': 5,
            'dropout_rate': 0.3,
            'hidden_dim': 256  # Smaller for faster testing
        }
        model = get_model(config)
        
        # 3. Create dataset
        print("3. Creating dataset...")
        dataset = SatelliteEconomicDataset(
            image_dir=str(Path(self.temp_dir) / 'images'),
            labels_file=str(Path(self.temp_dir) / 'labels.csv')
        )
        
        # 4. Test forward pass
        print("4. Testing forward pass...")
        image, labels = dataset[0]
        model.eval()
        
        with torch.no_grad():
            predictions = model(image.unsqueeze(0))
        
        # 5. Verify outputs
        print("5. Verifying outputs...")
        self.assertEqual(len(predictions), 5)
        for task, pred in predictions.items():
            self.assertEqual(pred.shape, (1,))
        
        print("✅ End-to-end test passed!")
        print("="*60)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestModel))
    suite.addTests(loader.loadTestsFromTestCase(TestLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
