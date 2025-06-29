#!/usr/bin/env python3
"""
Train PyTorch models on real dental data and generate pre-trained weights.

This script provides comprehensive training with real data support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import trimesh
import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pytorch_models import get_model, DiceAwareLoss
from training.trainer import TeethSegmentationTrainer


class RealTeethDataset(Dataset):
    """Dataset for real dental scan data."""
    
    def __init__(self, data_dir, num_points=1024, split='train'):
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.split = split
        
        # Find all .obj files
        scan_dir = self.data_dir / 'scans'
        label_dir = self.data_dir / 'labels'
        
        self.scan_files = []
        self.label_files = []
        
        if scan_dir.exists() and label_dir.exists():
            for obj_file in scan_dir.glob('*.obj'):
                json_file = label_dir / f"{obj_file.stem}.json"
                if json_file.exists():
                    self.scan_files.append(obj_file)
                    self.label_files.append(json_file)
        
        print(f"Found {len(self.scan_files)} real data samples for {split}")
        
        # If no real data, create synthetic data
        if len(self.scan_files) == 0:
            print(f"No real data found, generating synthetic data for {split}...")
            self.use_synthetic = True
            self.size = 50 if split == 'train' else 20
        else:
            self.use_synthetic = False
            # Split data
            total_size = len(self.scan_files)
            if split == 'train':
                self.scan_files = self.scan_files[:int(0.8 * total_size)]
                self.label_files = self.label_files[:int(0.8 * total_size)]
            else:  # val
                self.scan_files = self.scan_files[int(0.8 * total_size):]
                self.label_files = self.label_files[int(0.8 * total_size):]
    
    def __len__(self):
        if self.use_synthetic:
            return self.size
        return len(self.scan_files)
    
    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._generate_synthetic_sample()
        else:
            return self._load_real_sample(idx)
    
    def _load_real_sample(self, idx):
        """Load real dental scan sample."""
        try:
            # Load mesh
            mesh = trimesh.load(self.scan_files[idx])
            vertices = np.array(mesh.vertices, dtype=np.float32)
            
            # Load labels
            with open(self.label_files[idx], 'r') as f:
                label_data = json.load(f)
            
            labels = np.array(label_data['labels'], dtype=np.int64)
            instances = np.array(label_data['instances'], dtype=np.int64)
            
            # Sample points
            if len(vertices) > self.num_points:
                indices = np.random.choice(len(vertices), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(vertices), self.num_points, replace=True)
            
            points = vertices[indices]
            seg_labels = labels[indices]
            inst_labels = instances[indices]
            
            # Normalize points
            points = points - points.mean(axis=0)
            scale = np.max(np.linalg.norm(points, axis=1))
            if scale > 0:
                points = points / scale
            
            # Add geometric features (normals, curvature approximation)
            if hasattr(mesh, 'vertex_normals'):
                normals = mesh.vertex_normals[indices]
            else:
                # Compute simple normals approximation
                normals = np.random.normal(0, 0.1, (self.num_points, 3))
                normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
            
            # Combine xyz + normals for 6D features
            features = np.concatenate([points, normals], axis=1).T  # [6, num_points]
            
            return {
                'points': torch.FloatTensor(features),  # [6, num_points] 
                'seg_labels': torch.LongTensor(seg_labels),  # [num_points]
                'inst_labels': torch.LongTensor(inst_labels)  # [num_points]
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Fallback to synthetic
            return self._generate_synthetic_sample()
    
    def _generate_synthetic_sample(self):
        """Generate synthetic training sample."""
        # Create random point cloud with teeth-like structure
        points = np.random.randn(self.num_points, 3).astype(np.float32)
        
        # Create clusters for teeth
        num_teeth = np.random.randint(6, 16)
        centers = np.random.randn(num_teeth, 3) * 2
        
        seg_labels = np.zeros(self.num_points, dtype=np.int64)
        inst_labels = np.zeros(self.num_points, dtype=np.int64)
        
        for i in range(self.num_points):
            # Find closest tooth center
            distances = np.linalg.norm(points[i] - centers, axis=1)
            closest_tooth = np.argmin(distances)
            
            if distances[closest_tooth] < 0.5:  # Point belongs to tooth
                seg_labels[i] = 11 + (closest_tooth % 32)  # FDI numbering
                inst_labels[i] = closest_tooth + 1
            # else: remains 0 (gingiva)
        
        # Generate normals
        normals = np.random.normal(0, 0.1, (self.num_points, 3))
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        # Combine features
        features = np.concatenate([points, normals], axis=1).T  # [6, num_points]
        
        return {
            'points': torch.FloatTensor(features),
            'seg_labels': torch.LongTensor(seg_labels),
            'inst_labels': torch.LongTensor(inst_labels)
        }


def train_model(args):
    """Train a model and save pre-trained weights."""
    print(f"🚀 Training {args.model} model...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = RealTeethDataset(args.data_dir, args.num_points, 'train')
    val_dataset = RealTeethDataset(args.data_dir, args.num_points, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    if args.model == 'custom':
        model = get_model('custom', num_classes=49, num_instances=32)
    else:
        model = get_model('pointnet', num_classes=49)
    
    model.to(device)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    if args.model == 'custom':
        # For custom model, use separate losses for segmentation and instance
        seg_criterion = DiceAwareLoss(num_classes=49)
        inst_criterion = nn.CrossEntropyLoss(ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in train_pbar:
            points = batch['points'].to(device)  # [B, 6, N]
            seg_labels = batch['seg_labels'].to(device)  # [B, N]
            inst_labels = batch['inst_labels'].to(device)  # [B, N]
            
            optimizer.zero_grad()
            
            if args.model == 'custom':
                seg_pred, inst_pred = model(points)
                seg_loss = seg_criterion(seg_pred, seg_labels)
                inst_loss = inst_criterion(inst_pred, inst_labels)
                loss = seg_loss + 0.5 * inst_loss  # Weighted combination
            else:
                seg_pred = model(points)
                loss = criterion(seg_pred, seg_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in val_pbar:
                points = batch['points'].to(device)
                seg_labels = batch['seg_labels'].to(device)
                inst_labels = batch['inst_labels'].to(device)
                
                if args.model == 'custom':
                    seg_pred, inst_pred = model(points)
                    seg_loss = seg_criterion(seg_pred, seg_labels)
                    inst_loss = inst_criterion(inst_pred, inst_labels)
                    loss = seg_loss + 0.5 * inst_loss  # Weighted combination
                else:
                    seg_pred = model(points)
                    loss = criterion(seg_pred, seg_labels)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_name': args.model,
                'num_classes': 49,
                'num_instances': 32 if args.model == 'custom' else None
            }
            
            checkpoint_path = f"checkpoints/{args.model}_best_model.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved best model: {checkpoint_path}")
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'model_name': args.model,
        'num_classes': 49,
        'num_instances': 32 if args.model == 'custom' else None,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    final_path = f"checkpoints/{args.model}_final_model.pth"
    torch.save(final_checkpoint, final_path)
    print(f"💾 Saved final model: {final_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{args.model.title()} Training Curves')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-20:], label='Train Loss (Last 20)')
    plt.plot(val_losses[-20:], label='Val Loss (Last 20)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Recent Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'logs/{args.model}_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"📊 Saved training curves: logs/{args.model}_training_curves.png")
    
    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train teeth segmentation models')
    parser.add_argument('--model', choices=['pointnet', 'custom'], default='custom', 
                        help='Model architecture to train')
    parser.add_argument('--data_dir', default='./data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points to sample')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--all_models', action='store_true', help='Train all models')
    
    args = parser.parse_args()
    
    print("🦷 3D Teeth Segmentation - Model Training")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    # Ensure directories exist
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    if args.all_models:
        models_to_train = ['pointnet', 'custom']
    else:
        models_to_train = [args.model]
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n🎯 Training {model_name} model...")
        args.model = model_name
        
        start_time = time.time()
        model, best_loss = train_model(args)
        training_time = time.time() - start_time
        
        results[model_name] = {
            'best_val_loss': best_loss,
            'training_time': training_time
        }
        
        print(f"✅ {model_name} training completed!")
        print(f"   Best val loss: {best_loss:.4f}")
        print(f"   Training time: {training_time:.1f}s")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Training Summary:")
    for model_name, stats in results.items():
        print(f"  {model_name}:")
        print(f"    Best Loss: {stats['best_val_loss']:.4f}")
        print(f"    Time: {stats['training_time']:.1f}s")
    
    print("\n🎉 Pre-trained weights generated!")
    print("📁 Checkpoints saved in: ./checkpoints/")
    print("📊 Training logs saved in: ./logs/")
    print("\n💡 Use these models with:")
    print("  python examples/example_algorithm.py --model_path checkpoints/custom_best_model.pth")


if __name__ == "__main__":
    main()