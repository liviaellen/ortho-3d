#!/usr/bin/env python3
"""
Quick training script to generate pre-trained weights fast.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pytorch_models import get_model, DiceAwareLoss

def quick_train_model(model_name='custom', epochs=3):
    """Quick training with minimal setup."""
    print(f"🚀 Quick training {model_name} model for {epochs} epochs...")
    
    device = torch.device('cpu')  # Force CPU for speed
    
    # Create model
    if model_name == 'custom':
        model = get_model('custom', num_classes=49, num_instances=32)
    else:
        model = get_model('pointnet', num_classes=49)
    
    model.to(device)
    
    # Simple training setup
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher LR for quick training
    
    if model_name == 'custom':
        seg_criterion = nn.CrossEntropyLoss(ignore_index=0)
        inst_criterion = nn.CrossEntropyLoss(ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Training with synthetic data...")
    
    # Quick training loop with synthetic data
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 5  # Quick training
        
        for batch_idx in range(num_batches):
            # Generate simple synthetic batch
            batch_size = 2
            num_points = 512  # Smaller for speed
            
            if model_name == 'custom':
                # 6D input for custom model
                points = torch.randn(batch_size, 6, num_points)
            else:
                # 3D input for pointnet
                points = torch.randn(batch_size, 3, num_points)
            
            # Simple labels
            seg_labels = torch.randint(0, 49, (batch_size, num_points))
            inst_labels = torch.randint(0, 32, (batch_size, num_points))
            
            optimizer.zero_grad()
            
            if model_name == 'custom':
                seg_pred, inst_pred = model(points)
                seg_loss = seg_criterion(seg_pred, seg_labels)
                inst_loss = inst_criterion(inst_pred, inst_labels)
                loss = seg_loss + 0.5 * inst_loss
            else:
                seg_pred = model(points)
                loss = criterion(seg_pred, seg_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # Save trained model
    Path("checkpoints").mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'num_classes': 49,
        'num_instances': 32 if model_name == 'custom' else None,
        'training_type': 'quick_synthetic'
    }
    
    checkpoint_path = f"checkpoints/{model_name}_quick_trained.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Saved model: {checkpoint_path}")
    
    return checkpoint_path

def test_trained_model(checkpoint_path):
    """Test the trained model."""
    print(f"🧪 Testing trained model: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_name = checkpoint['model_name']
    
    # Create model
    if model_name == 'custom':
        model = get_model('custom', num_classes=49, num_instances=32)
    else:
        model = get_model('pointnet', num_classes=49)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test inference
    with torch.no_grad():
        if model_name == 'custom':
            test_input = torch.randn(1, 6, 1024)
            seg_out, inst_out = model(test_input)
            print(f"  ✅ Custom model working: seg={seg_out.shape}, inst={inst_out.shape}")
        else:
            test_input = torch.randn(1, 3, 1024)
            output = model(test_input)
            print(f"  ✅ PointNet model working: output={output.shape}")
    
    print(f"  📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def main():
    """Train both models quickly."""
    print("🦷 Quick Pre-trained Weights Generation")
    print("=" * 50)
    
    models_to_train = ['pointnet', 'custom']
    trained_models = []
    
    for model_name in models_to_train:
        print(f"\n🎯 Training {model_name}...")
        checkpoint_path = quick_train_model(model_name, epochs=3)
        trained_models.append(checkpoint_path)
        
        # Test the model
        test_trained_model(checkpoint_path)
    
    print("\n" + "=" * 50)
    print("✅ Quick training completed!")
    print("\n📁 Pre-trained weights available:")
    for path in trained_models:
        print(f"  - {path}")
    
    print(f"\n💡 Use these models with:")
    print(f"  python examples/example_algorithm.py --model_path {trained_models[-1]}")
    print(f"  python launch_ui.py")

if __name__ == "__main__":
    main()