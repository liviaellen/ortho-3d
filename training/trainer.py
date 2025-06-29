#!/usr/bin/env python3
"""
PyTorch Training Pipeline for 3D Teeth Segmentation

This module provides a comprehensive training pipeline for 3D teeth segmentation
including data loading, model training, validation, and checkpointing.

Author: Enhanced for academic research with PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Any
import logging
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.pytorch_models import get_model, DiceAwareLoss


class TeethSegmentationTrainer:
    """
    Comprehensive trainer for 3D teeth segmentation models.
    """
    
    def __init__(
        self,
        model_name: str = 'pointnet',
        num_classes: int = 49,
        num_instances: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        save_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of segmentation classes
            num_instances: Number of instance classes
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            device: Device to train on
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Create directories
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            num_instances=num_instances
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Initialize loss function
        self.criterion = DiceAwareLoss(num_classes=num_classes)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_inst_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            points = batch['points'].to(self.device)
            seg_labels = batch['seg_labels'].to(self.device)
            inst_labels = batch['inst_labels'].to(self.device)
            
            batch_size = points.size(0)
            total_samples += batch_size
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_name == 'custom':
                seg_logits, inst_logits = self.model(points)
                
                # Compute losses
                seg_loss = self.criterion(seg_logits, seg_labels)
                inst_loss = F.cross_entropy(inst_logits, inst_labels, ignore_index=-1)
                loss = seg_loss + 0.5 * inst_loss
                
                total_seg_loss += seg_loss.item() * batch_size
                total_inst_loss += inst_loss.item() * batch_size
            else:
                seg_logits = self.model(points)
                loss = self.criterion(seg_logits, seg_labels)
                total_seg_loss += loss.item() * batch_size
            
            total_loss += loss.item() * batch_size
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate average losses
        avg_loss = total_loss / total_samples
        avg_seg_loss = total_seg_loss / total_samples
        avg_inst_loss = total_inst_loss / total_samples if self.model_name == 'custom' else 0.0
        
        metrics = {
            'loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'inst_loss': avg_inst_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_inst_loss = 0.0
        total_samples = 0
        
        # Metrics for evaluation
        total_iou = 0.0
        total_dice = 0.0
        total_accuracy = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                points = batch['points'].to(self.device)
                seg_labels = batch['seg_labels'].to(self.device)
                inst_labels = batch['inst_labels'].to(self.device)
                
                batch_size = points.size(0)
                total_samples += batch_size
                
                # Forward pass
                if self.model_name == 'custom':
                    seg_logits, inst_logits = self.model(points)
                    
                    # Compute losses
                    seg_loss = self.criterion(seg_logits, seg_labels)
                    inst_loss = F.cross_entropy(inst_logits, inst_labels, ignore_index=-1)
                    loss = seg_loss + 0.5 * inst_loss
                    
                    total_seg_loss += seg_loss.item() * batch_size
                    total_inst_loss += inst_loss.item() * batch_size
                else:
                    seg_logits = self.model(points)
                    loss = self.criterion(seg_logits, seg_labels)
                    total_seg_loss += loss.item() * batch_size
                
                total_loss += loss.item() * batch_size
                
                # Calculate metrics
                seg_pred = torch.argmax(seg_logits, dim=1)
                
                # IoU calculation (simplified)
                iou = self._calculate_iou(seg_pred, seg_labels)
                dice = self._calculate_dice(seg_pred, seg_labels)
                accuracy = (seg_pred == seg_labels).float().mean()
                
                total_iou += iou * batch_size
                total_dice += dice * batch_size
                total_accuracy += accuracy * batch_size
        
        # Calculate average metrics
        avg_loss = total_loss / total_samples
        avg_seg_loss = total_seg_loss / total_samples
        avg_inst_loss = total_inst_loss / total_samples if self.model_name == 'custom' else 0.0
        avg_iou = total_iou / total_samples
        avg_dice = total_dice / total_samples
        avg_accuracy = total_accuracy / total_samples
        
        metrics = {
            'loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'inst_loss': avg_inst_loss,
            'iou': avg_iou,
            'dice': avg_dice,
            'accuracy': avg_accuracy
        }
        
        return metrics
    
    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate IoU score."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Exclude background class (0)
        mask = target_flat != 0
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
        
        if len(target_flat) == 0:
            return 0.0
        
        intersection = (pred_flat == target_flat).sum().float()
        union = len(target_flat)
        
        return (intersection / union).item()
    
    def _calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice coefficient."""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Exclude background class (0)
        mask = target_flat != 0
        pred_flat = pred_flat[mask]
        target_flat = target_flat[mask]
        
        if len(target_flat) == 0:
            return 0.0
        
        intersection = (pred_flat == target_flat).sum().float()
        total = len(pred_flat) + len(target_flat)
        
        return (2.0 * intersection / total).item()
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics_history': self.metrics_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved with IoU: {metrics["iou"]:.4f}')
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_val_iou = checkpoint['best_val_iou']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.metrics_history = checkpoint.get('metrics_history', {})
            
            self.logger.info(f'Loaded checkpoint from epoch {self.current_epoch}')
            return True
        
        except Exception as e:
            self.logger.error(f'Failed to load checkpoint: {e}')
            return False
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        save_every: int = 10,
        early_stopping_patience: int = 20
    ):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Early stopping patience
        """
        self.logger.info(f'Starting training for {num_epochs} epochs')
        self.logger.info(f'Model: {self.model_name}, Device: {self.device}')
        
        no_improvement_count = 0
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            epoch_time = time.time() - start_time
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            
            # Store metrics history
            for key, value in train_metrics.items():
                if f'train_{key}' not in self.metrics_history:
                    self.metrics_history[f'train_{key}'] = []
                self.metrics_history[f'train_{key}'].append(value)
            
            for key, value in val_metrics.items():
                if f'val_{key}' not in self.metrics_history:
                    self.metrics_history[f'val_{key}'] = []
                self.metrics_history[f'val_{key}'].append(value)
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('IoU/Val', val_metrics['iou'], epoch)
            self.writer.add_scalar('Dice/Val', val_metrics['dice'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', train_metrics['learning_rate'], epoch)
            
            # Check for best model
            is_best = val_metrics['iou'] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics['iou']
                self.best_val_loss = val_metrics['loss']
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Log epoch results
            self.logger.info(
                f'Epoch {epoch:3d}/{num_epochs} ({epoch_time:.1f}s) - '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Val Loss: {val_metrics["loss"]:.4f}, '
                f'Val IoU: {val_metrics["iou"]:.4f}, '
                f'Val Dice: {val_metrics["dice"]:.4f}'
            )
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                self.logger.info(f'Early stopping at epoch {epoch}')
                break
        
        self.logger.info('Training completed!')
        self.logger.info(f'Best validation IoU: {self.best_val_iou:.4f}')
        
        # Close tensorboard writer
        self.writer.close()


def main():
    """Main function for testing the trainer."""
    # This would typically load real data
    # For now, we'll create dummy data loaders
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'points': torch.randn(6, 1024),  # 6D features, 1024 points
                'seg_labels': torch.randint(0, 49, (1024,)),
                'inst_labels': torch.randint(0, 32, (1024,))
            }
    
    # Create data loaders
    train_dataset = DummyDataset(size=100)
    val_dataset = DummyDataset(size=20)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize trainer
    trainer = TeethSegmentationTrainer(
        model_name='custom',
        num_classes=49,
        learning_rate=0.001
    )
    
    # Train for a few epochs
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        save_every=2
    )


if __name__ == "__main__":
    main()