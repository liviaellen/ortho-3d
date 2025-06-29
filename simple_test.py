#!/usr/bin/env python3
"""
Simple test script for PyTorch 3D Teeth Segmentation
"""

import torch
import numpy as np
import sys
import os

print("🦷 Simple PyTorch Test")
print("=" * 30)

# Test 1: Basic PyTorch
print("1. Testing PyTorch...")
try:
    x = torch.randn(2, 3, 1024)
    print(f"✅ PyTorch working! Tensor shape: {x.shape}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch test failed: {e}")

# Test 2: Models
print("\n2. Testing Models...")
try:
    from models.pytorch_models import PointNetSegmentation
    model = PointNetSegmentation(num_classes=49)
    x = torch.randn(1, 3, 1024)
    output = model(x)
    print(f"✅ PointNet model working! Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Model test failed: {e}")

# Test 3: Custom Model
print("\n3. Testing Custom Model...")
try:
    from models.pytorch_models import TeethSegmentationNet
    model = TeethSegmentationNet(num_classes=49, num_instances=32)
    x = torch.randn(1, 6, 1024)  # 6D input (xyz + normals)
    seg_out, inst_out = model(x)
    print(f"✅ Custom model working!")
    print(f"   Segmentation output: {seg_out.shape}")
    print(f"   Instance output: {inst_out.shape}")
except Exception as e:
    print(f"❌ Custom model test failed: {e}")

# Test 4: Simple algorithm
print("\n4. Testing Simple Algorithm...")
try:
    from examples.example_algorithm import PyTorchSegmentationAlgorithm
    import trimesh
    
    # Create dummy mesh
    vertices = np.random.rand(500, 3) * 10
    faces = np.random.randint(0, 500, (250, 3))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh_path = "simple_test_mesh.obj"
    mesh.export(mesh_path)
    
    # Test algorithm
    algorithm = PyTorchSegmentationAlgorithm(
        model_name='custom',
        num_points=256
    )
    
    labels, instances = algorithm.process(mesh_path)
    
    print(f"✅ Algorithm working!")
    print(f"   Processed {len(labels)} vertices")
    print(f"   Found {len(np.unique(instances))} instances")
    
    # Clean up
    if os.path.exists(mesh_path):
        os.remove(mesh_path)
        
except Exception as e:
    print(f"❌ Algorithm test failed: {e}")

# Test 5: Basic evaluation
print("\n5. Testing Basic Evaluation...")
try:
    sys.path.append('evaluation')
    from evaluation.evaluation import calculate_metrics
    
    # Create simple dummy data
    n_verts = 100
    dummy_gt = {
        'mesh_vertices': np.random.rand(n_verts, 3),
        'instances': np.random.randint(0, 5, n_verts),
        'labels': np.random.randint(11, 20, n_verts)
    }
    
    dummy_pred = {
        'instances': np.random.randint(0, 5, n_verts),
        'labels': np.random.randint(11, 20, n_verts)
    }
    
    # This should work even if some parts fail
    print("   Calculating basic metrics...")
    results = calculate_metrics(dummy_gt, dummy_pred)
    print(f"✅ Basic evaluation working!")
    print(f"   Got {len(results)} metric values")
    
except Exception as e:
    print(f"❌ Evaluation test failed: {e}")

print("\n🎉 Simple test completed!")
print("\nIf most tests passed, your setup is working!")
print("You can now try:")
print("  - python examples/example_algorithm.py")
print("  - python scripts/train_pytorch.py --epochs 1")