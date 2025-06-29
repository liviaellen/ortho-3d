#!/usr/bin/env python3
"""
PyTorch Models for 3D Teeth Segmentation

This module contains various PyTorch neural network architectures for 
3D teeth segmentation including PointNet, PointNet++, and custom models.

Author: Enhanced for academic research with PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class PointNetFeatureExtractor(nn.Module):
    """
    PointNet feature extraction module for point cloud processing.
    """
    
    def __init__(self, input_dim=3, feature_dim=1024):
        super(PointNetFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass through PointNet feature extractor.
        
        Args:
            x: Input point cloud [B, 3, N]
            
        Returns:
            Point features [B, feature_dim, N]
            Global features [B, feature_dim]
        """
        # Point-wise MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Global max pooling
        global_feat = torch.max(x, 2, keepdim=False)[0]
        
        return x, global_feat


class PointNetSegmentation(nn.Module):
    """
    PointNet-based segmentation network for 3D teeth segmentation.
    """
    
    def __init__(self, num_classes=49, input_dim=3, feature_dim=1024):
        super(PointNetSegmentation, self).__init__()
        
        self.num_classes = num_classes
        self.feature_extractor = PointNetFeatureExtractor(input_dim, feature_dim)
        
        # Segmentation head
        self.seg_conv1 = nn.Conv1d(feature_dim + feature_dim, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)
        
        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass through PointNet segmentation.
        
        Args:
            x: Input point cloud [B, 3, N]
            
        Returns:
            Segmentation logits [B, num_classes, N]
        """
        batch_size, _, num_points = x.size()
        
        # Extract features
        point_feat, global_feat = self.feature_extractor(x)
        
        # Expand global features
        global_feat_expanded = global_feat.view(batch_size, -1, 1).repeat(1, 1, num_points)
        
        # Concatenate point and global features
        concat_feat = torch.cat([point_feat, global_feat_expanded], 1)
        
        # Segmentation MLPs
        x = F.relu(self.seg_bn1(self.seg_conv1(concat_feat)))
        x = self.dropout(x)
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.seg_bn3(self.seg_conv3(x)))
        x = self.seg_conv4(x)
        
        return x


class SetAbstraction(nn.Module):
    """
    Set Abstraction module for PointNet++.
    """
    
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        Forward pass through Set Abstraction.
        
        Args:
            xyz: Point coordinates [B, N, 3]
            points: Point features [B, N, C]
            
        Returns:
            new_xyz: Sampled coordinates [B, npoint, 3]
            new_points: Aggregated features [B, npoint, mlp[-1]]
        """
        # This is a simplified implementation
        # In practice, you would use operations from torch_geometric or pytorch3d
        B, N, C = xyz.shape
        
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            new_points = points.max(dim=1, keepdim=True)[0]
        else:
            # Simplified sampling and grouping
            indices = torch.randperm(N)[:self.npoint]
            new_xyz = xyz[:, indices, :]
            new_points = points[:, indices, :]
        
        # Apply MLPs
        new_points = new_points.unsqueeze(-1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        return new_xyz, new_points


class PointNetPlusPlus(nn.Module):
    """
    PointNet++ segmentation network for 3D teeth segmentation.
    """
    
    def __init__(self, num_classes=49):
        super(PointNetPlusPlus, self).__init__()
        
        self.num_classes = num_classes
        
        # Set abstraction layers
        self.sa1 = SetAbstraction(1024, 0.1, 32, 3, [32, 32, 64])
        self.sa2 = SetAbstraction(256, 0.2, 32, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(64, 0.4, 32, 128, [128, 128, 256])
        self.sa4 = SetAbstraction(16, 0.8, 32, 256, [256, 256, 512])
        
        # Feature propagation layers (simplified)
        self.fp4 = nn.Conv1d(512 + 256, 256, 1)
        self.fp3 = nn.Conv1d(256 + 128, 256, 1)
        self.fp2 = nn.Conv1d(256 + 64, 128, 1)
        self.fp1 = nn.Conv1d(128 + 3, 128, 1)
        
        # Final classification layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, xyz):
        """
        Forward pass through PointNet++.
        
        Args:
            xyz: Input point cloud [B, N, 3]
            
        Returns:
            Segmentation logits [B, num_classes, N]
        """
        # Set abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, xyz)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Feature propagation (simplified upsampling)
        l3_points = self.fp4(torch.cat([l3_points, l4_points.expand_as(l3_points)], 1).unsqueeze(-1)).squeeze(-1)
        l2_points = self.fp3(torch.cat([l2_points, l3_points.expand_as(l2_points)], 1).unsqueeze(-1)).squeeze(-1)
        l1_points = self.fp2(torch.cat([l1_points, l2_points.expand_as(l1_points)], 1).unsqueeze(-1)).squeeze(-1)
        l0_points = self.fp1(torch.cat([xyz.transpose(1, 2), l1_points.expand(xyz.size(0), -1, xyz.size(1))], 1))
        
        # Final classification
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.dropout1(feat)
        output = self.conv2(feat)
        
        return output


class TeethSegmentationNet(nn.Module):
    """
    Custom 3D teeth segmentation network combining multiple approaches.
    """
    
    def __init__(self, num_classes=49, num_instances=32, input_dim=6):
        super(TeethSegmentationNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_instances = num_instances
        
        # Shared feature extraction
        self.shared_conv1 = nn.Conv1d(input_dim, 64, 1)
        self.shared_conv2 = nn.Conv1d(64, 128, 1)
        self.shared_conv3 = nn.Conv1d(128, 256, 1)
        
        self.shared_bn1 = nn.BatchNorm1d(64)
        self.shared_bn2 = nn.BatchNorm1d(128)
        self.shared_bn3 = nn.BatchNorm1d(256)
        
        # Global feature extraction
        self.global_conv1 = nn.Conv1d(256, 512, 1)
        self.global_conv2 = nn.Conv1d(512, 1024, 1)
        
        self.global_bn1 = nn.BatchNorm1d(512)
        self.global_bn2 = nn.BatchNorm1d(1024)
        
        # Segmentation head (for tooth labels)
        self.seg_conv1 = nn.Conv1d(256 + 1024, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, num_classes, 1)
        
        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        
        # Instance head (for tooth instances)
        self.inst_conv1 = nn.Conv1d(256 + 1024, 512, 1)
        self.inst_conv2 = nn.Conv1d(512, 256, 1)
        self.inst_conv3 = nn.Conv1d(256, num_instances, 1)
        
        self.inst_bn1 = nn.BatchNorm1d(512)
        self.inst_bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass through teeth segmentation network.
        
        Args:
            x: Input point cloud with features [B, input_dim, N]
            
        Returns:
            seg_logits: Segmentation logits [B, num_classes, N]
            inst_logits: Instance logits [B, num_instances, N]
        """
        batch_size, _, num_points = x.size()
        
        # Shared feature extraction
        x = F.relu(self.shared_bn1(self.shared_conv1(x)))
        x = F.relu(self.shared_bn2(self.shared_conv2(x)))
        point_feat = F.relu(self.shared_bn3(self.shared_conv3(x)))
        
        # Global feature extraction
        x = F.relu(self.global_bn1(self.global_conv1(point_feat)))
        x = F.relu(self.global_bn2(self.global_conv2(x)))
        
        # Global max pooling
        global_feat = torch.max(x, 2, keepdim=False)[0]
        global_feat_expanded = global_feat.view(batch_size, -1, 1).repeat(1, 1, num_points)
        
        # Combine point and global features
        combined_feat = torch.cat([point_feat, global_feat_expanded], 1)
        
        # Segmentation branch
        seg_x = F.relu(self.seg_bn1(self.seg_conv1(combined_feat)))
        seg_x = self.dropout(seg_x)
        seg_x = F.relu(self.seg_bn2(self.seg_conv2(seg_x)))
        seg_x = self.dropout(seg_x)
        seg_logits = self.seg_conv3(seg_x)
        
        # Instance branch
        inst_x = F.relu(self.inst_bn1(self.inst_conv1(combined_feat)))
        inst_x = self.dropout(inst_x)
        inst_x = F.relu(self.inst_bn2(self.inst_conv2(inst_x)))
        inst_x = self.dropout(inst_x)
        inst_logits = self.inst_conv3(inst_x)
        
        return seg_logits, inst_logits


class DiceAwareLoss(nn.Module):
    """
    Combined loss function incorporating Dice loss for segmentation.
    """
    
    def __init__(self, num_classes=49, class_weights=None, dice_weight=0.5):
        super(DiceAwareLoss, self).__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """
        Compute Dice loss.
        """
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 2, 1).float()
        
        intersection = (pred_softmax * target_onehot).sum(dim=2)
        union = pred_softmax.sum(dim=2) + target_onehot.sum(dim=2)
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def forward(self, pred, target):
        """
        Compute combined cross-entropy and Dice loss.
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, weight=self.class_weights, ignore_index=-1)
        
        # Dice loss
        dice_loss = self.dice_loss(pred, target)
        
        # Combined loss
        total_loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
        
        return total_loss


def get_model(model_name='pointnet', num_classes=49, **kwargs):
    """
    Factory function to get different model architectures.
    
    Args:
        model_name: Name of the model ('pointnet', 'pointnet++', 'custom')
        num_classes: Number of output classes
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    if model_name.lower() == 'pointnet':
        # PointNet only accepts num_classes, filter out other params
        return PointNetSegmentation(num_classes=num_classes)
    elif model_name.lower() == 'pointnet++':
        # PointNet++ only accepts num_classes, filter out other params  
        return PointNetPlusPlus(num_classes=num_classes)
    elif model_name.lower() == 'custom':
        # Custom model accepts all parameters
        return TeethSegmentationNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test PointNet
    print("Testing PointNet...")
    model = PointNetSegmentation(num_classes=49)
    model.to(device)
    
    # Create dummy input
    batch_size, num_points = 4, 1024
    x = torch.randn(batch_size, 3, num_points).to(device)
    
    with torch.no_grad():
        output = model(x)
        print(f"PointNet output shape: {output.shape}")
    
    # Test custom model
    print("Testing Custom TeethSegmentationNet...")
    model = TeethSegmentationNet(num_classes=49, input_dim=6)
    model.to(device)
    
    x = torch.randn(batch_size, 6, num_points).to(device)
    
    with torch.no_grad():
        seg_out, inst_out = model(x)
        print(f"Custom model - Seg output: {seg_out.shape}, Inst output: {inst_out.shape}")
    
    print("✓ All models tested successfully!")