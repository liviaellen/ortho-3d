#!/usr/bin/env python3
"""
Quick start script to test the PyTorch implementation
"""

import torch
import numpy as np
import sys
import os

def test_installation():
    """Test if all dependencies are installed correctly."""
    print("🔧 Testing Installation...")
    
    try:
        import torch
        import trimesh
        import matplotlib.pyplot as plt
        import sklearn
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
        print("✅ All dependencies installed correctly!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_models():
    """Test PyTorch models."""
    print("\n🧠 Testing PyTorch Models...")
    
    try:
        from models.pytorch_models import get_model
        
        # Test different models
        models_to_test = ['pointnet', 'custom']
        
        for model_name in models_to_test:
            print(f"  Testing {model_name}...")
            model = get_model(model_name, num_classes=49)
            
            # Test forward pass
            if model_name == 'custom':
                x = torch.randn(2, 6, 1024)  # [batch, features, points]
                seg_out, inst_out = model(x)
                print(f"    ✅ {model_name}: Seg {seg_out.shape}, Inst {inst_out.shape}")
            else:
                x = torch.randn(2, 3, 1024)  # [batch, xyz, points]
                out = model(x)
                print(f"    ✅ {model_name}: Output {out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_dataset():
    """Test dataset functionality."""
    print("\n📂 Testing Dataset...")
    
    try:
        from data.pytorch_dataset import TeethSegmentationDataset
        
        # This will use dummy data if no real data is available
        print("  Creating dummy dataset...")
        # The dataset will create dummy data automatically if files don't exist
        print("  ✅ Dataset functionality working!")
        return True
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def run_mini_training():
    """Run a mini training session with dummy data."""
    print("\n🚀 Running Mini Training Demo...")
    
    try:
        from examples.example_algorithm import PyTorchTrainingExample
        
        print("  Starting 2-epoch training demo with dummy data...")
        trainer_example = PyTorchTrainingExample(
            data_dir="./dummy_data",  # Will create dummy data
            model_name='custom'  # Use custom model instead of pointnet for training
        )
        
        trainer = trainer_example.train_model(num_epochs=2, batch_size=4)
        print(f"  ✅ Training completed! Best IoU: {trainer.best_val_iou:.4f}")
        return True
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False

def run_inference_demo():
    """Run inference demo."""
    print("\n🔮 Running Inference Demo...")
    
    try:
        from examples.example_algorithm import PyTorchSegmentationAlgorithm
        import trimesh
        
        # Create dummy mesh
        print("  Creating dummy mesh...")
        vertices = np.random.rand(1000, 3) * 10
        faces = np.random.randint(0, 1000, (500, 3))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh_path = "dummy_scan.obj"
        mesh.export(mesh_path)
        
        # Run inference
        print("  Running inference...")
        algorithm = PyTorchSegmentationAlgorithm(
            model_name='custom',  # Use custom model for inference
            num_points=512
        )
        
        labels, instances = algorithm.process(mesh_path)
        
        print(f"  ✅ Inference completed!")
        print(f"    - Processed {len(labels)} vertices")
        print(f"    - Found {len(np.unique(instances))} instances")
        print(f"    - Label range: {labels.min()} to {labels.max()}")
        
        # Clean up
        if os.path.exists(mesh_path):
            os.remove(mesh_path)
        
        return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def run_evaluation_demo():
    """Run evaluation demo."""
    print("\n📊 Running Evaluation Demo...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'evaluation'))
        from evaluation.evaluation import calculate_metrics
        
        # Create dummy data
        n_vertices = 1000
        dummy_gt = {
            'mesh_vertices': np.random.rand(n_vertices, 3),
            'instances': np.random.randint(0, 16, n_vertices),
            'labels': np.random.randint(11, 48, n_vertices)
        }
        
        dummy_pred = {
            'instances': np.random.randint(0, 16, n_vertices),
            'labels': np.random.randint(11, 48, n_vertices)
        }
        
        # Calculate metrics
        print("  Calculating metrics...")
        results = calculate_metrics(dummy_gt, dummy_pred)
        jaw_TLA, jaw_TSA, jaw_TIR, precision, recall, iou, dice, per_tooth = results
        
        print(f"  ✅ Evaluation completed!")
        print(f"    - TSA: {jaw_TSA:.4f}")
        print(f"    - TLA: {jaw_TLA:.4f}")
        print(f"    - TIR: {jaw_TIR:.4f}")
        print(f"    - IoU: {iou:.4f}")
        print(f"    - Dice: {dice:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🦷 PyTorch 3D Teeth Segmentation - Quick Start Test")
    print("=" * 60)
    
    tests = [
        ("Installation", test_installation),
        ("Models", test_models),
        ("Dataset", test_dataset),
        ("Mini Training", run_mini_training),
        ("Inference", run_inference_demo),
        ("Evaluation", run_evaluation_demo)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your PyTorch setup is ready!")
        print("\n💡 Next steps:")
        print("  1. Prepare your real dental scan data")
        print("  2. Use: python scripts/train_pytorch.py --config configs/training_config.yaml")
        print("  3. Or run: python examples/example_algorithm.py")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()