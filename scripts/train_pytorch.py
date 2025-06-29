#!/usr/bin/env python3
"""
Training Script for PyTorch 3D Teeth Segmentation Models

This script provides a command-line interface for training PyTorch models
using configuration files.

Usage:
    python scripts/train_pytorch.py --config configs/training_config.yaml
    python scripts/train_pytorch.py --model pointnet --epochs 50 --lr 0.001

Author: Enhanced for academic research with PyTorch
"""

import argparse
import yaml
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from training.trainer import TeethSegmentationTrainer
from data.pytorch_dataset import TeethSegmentationDataModule
from torch.utils.data import DataLoader


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_data_loaders(config):
    """Setup data loaders based on configuration."""
    data_config = config['data']
    
    data_module = TeethSegmentationDataModule(
        data_dir=data_config['data_dir'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        num_points=data_config['num_points'],
        augment=data_config['augment'],
        pin_memory=data_config['pin_memory'],
        normalize=data_config['normalize'],
        use_normals=data_config['use_normals'],
        use_colors=data_config['use_colors'],
        cache_data=data_config['cache_data']
    )
    
    data_module.setup()
    
    return data_module.train_dataloader(), data_module.val_dataloader()


def setup_trainer(config):
    """Setup trainer based on configuration."""
    model_config = config['model']
    training_config = config['training']
    paths_config = config['paths']
    
    # Determine device
    device_config = config['device']
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    
    trainer = TeethSegmentationTrainer(
        model_name=model_config['name'],
        num_classes=model_config['num_classes'],
        num_instances=model_config['num_instances'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        device=device,
        save_dir=paths_config['save_dir'],
        log_dir=paths_config['log_dir']
    )
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train 3D Teeth Segmentation Models')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to configuration file')
    
    # Override options
    parser.add_argument('--model', type=str, choices=['pointnet', 'pointnet++', 'custom'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Device options
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                        default='auto', help='Device to use for training')
    
    # Logging options
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable tensorboard logging')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='teeth_segmentation',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"✓ Loaded configuration from {args.config}")
    else:
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.save_dir:
        config['paths']['save_dir'] = args.save_dir
    if args.device != 'auto':
        config['device'] = args.device
    
    # Print configuration
    print("\n📋 Training Configuration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Batch Size: {config['data']['batch_size']}")
    print(f"  Data Directory: {config['data']['data_dir']}")
    print(f"  Device: {config['device']}")
    
    try:
        # Setup data loaders
        print("\n📂 Setting up data loaders...")
        train_loader, val_loader = setup_data_loaders(config)
        print(f"✓ Train samples: {len(train_loader.dataset)}")
        print(f"✓ Validation samples: {len(val_loader.dataset)}")
        
        # Setup trainer
        print("\n🔧 Setting up trainer...")
        trainer = setup_trainer(config)
        print(f"✓ Trainer initialized with {config['model']['name']} model")
        print(f"✓ Training on device: {trainer.device}")
        
        # Resume from checkpoint if specified
        if args.resume:
            if trainer.load_checkpoint(args.resume):
                print(f"✓ Resumed from checkpoint: {args.resume}")
            else:
                print(f"❌ Failed to resume from checkpoint: {args.resume}")
                sys.exit(1)
        
        # Start training
        print(f"\n🚀 Starting training for {config['training']['num_epochs']} epochs...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            save_every=config['training']['save_every'],
            early_stopping_patience=config['training']['early_stopping_patience']
        )
        
        print(f"\n🎉 Training completed!")
        print(f"✓ Best validation IoU: {trainer.best_val_iou:.4f}")
        print(f"✓ Best model saved to: {trainer.save_dir / 'best_model.pth'}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()