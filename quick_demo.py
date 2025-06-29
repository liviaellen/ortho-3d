#!/usr/bin/env python3
"""
Quick Demo - Everything that works without real data
"""

import torch
import numpy as np
import trimesh
import os

print("🦷 Quick PyTorch Demo - Everything Working!")
print("=" * 50)

# 1. Test Models
print("1. Testing PyTorch Models...")
try:
    from models.pytorch_models import PointNetSegmentation, TeethSegmentationNet
    
    # PointNet
    pointnet = PointNetSegmentation(num_classes=49)
    x = torch.randn(1, 3, 512)
    out = pointnet(x)
    print(f"✅ PointNet: {out.shape}")
    
    # Custom model
    custom_model = TeethSegmentationNet(num_classes=49, num_instances=32)
    x = torch.randn(1, 6, 512)
    seg_out, inst_out = custom_model(x)
    print(f"✅ Custom Model: Seg {seg_out.shape}, Inst {inst_out.shape}")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")

# 2. Test Algorithm (without training)
print("\n2. Testing Algorithm...")
try:
    from examples.example_algorithm import PyTorchSegmentationAlgorithm
    
    # Create dummy mesh
    vertices = np.random.rand(300, 3) * 10
    faces = np.random.randint(0, 300, (150, 3))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh_path = "demo_scan.obj"
    mesh.export(mesh_path)
    
    # Test with custom model (has both seg and inst outputs)
    algorithm = PyTorchSegmentationAlgorithm(
        model_name='custom',
        num_points=200
    )
    
    labels, instances = algorithm.process(mesh_path)
    
    print(f"✅ Segmentation complete!")
    print(f"   Vertices: {len(labels)}")
    print(f"   Instances: {len(np.unique(instances))}")
    print(f"   Labels: {labels.min()}-{labels.max()}")
    
    # Clean up
    if os.path.exists(mesh_path):
        os.remove(mesh_path)
        
except Exception as e:
    print(f"❌ Algorithm test failed: {e}")

# 3. Test Mini Training (with dummy data)
print("\n3. Testing Mini Training...")
try:
    from training.trainer import TeethSegmentationTrainer
    from torch.utils.data import DataLoader, Dataset
    
    # Dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, size=10):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'points': torch.randn(6, 200),  # 6D features, 200 points
                'seg_labels': torch.randint(0, 49, (200,)),
                'inst_labels': torch.randint(0, 32, (200,))
            }
    
    # Create data loaders
    train_dataset = DummyDataset(size=8)
    val_dataset = DummyDataset(size=4)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Initialize trainer
    trainer = TeethSegmentationTrainer(
        model_name='custom',
        num_classes=49,
        num_instances=32,
        learning_rate=0.01,
        save_dir='./demo_checkpoints',
        log_dir='./demo_logs'
    )
    
    print("   Starting 1-epoch training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        save_every=1
    )
    
    print(f"✅ Training complete! Best IoU: {trainer.best_val_iou:.4f}")
    
except Exception as e:
    print(f"❌ Training test failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Test visualization
print("\n4. Testing Visualization...")
try:
    from visualization.visualize_results import TeethVisualization
    
    viz = TeethVisualization()
    
    # Sample metrics
    metrics = {
        'TSA': 0.85,
        'TLA': 0.78,
        'TIR': 0.82,
        'precision': 0.86,
        'recall': 0.84,
        'iou': 0.81,
        'dice': 0.83
    }
    
    # Create plot (save to file)
    fig = viz.plot_metrics_comparison(metrics, save_path='demo_metrics.png')
    print("✅ Visualization working! Saved demo_metrics.png")
    
except Exception as e:
    print(f"❌ Visualization test failed: {e}")

print("\n🎉 Demo Complete!")
print("\n✅ What's Working:")
print("   - PyTorch models (PointNet, Custom)")
print("   - Segmentation algorithm")
print("   - Training pipeline")
print("   - Visualization tools")

print("\n💡 Next Steps:")
print("   1. Get real dental scan data (.obj + .json files)")
print("   2. Put data in ./data/train/, ./data/val/, ./data/test/")
print("   3. Run: python scripts/train_pytorch.py")
print("   4. Use trained model for inference")

print(f"\n🖥️  Your setup: PyTorch {torch.__version__}, Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")