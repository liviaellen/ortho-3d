#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader for 3D Teeth Segmentation

This module provides PyTorch dataset classes and data loading utilities
for the 3D teeth segmentation challenge.

Author: Enhanced for academic research with PyTorch
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import trimesh
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import random
from sklearn.preprocessing import StandardScaler
import logging


class TeethSegmentationDataset(Dataset):
    """
    PyTorch Dataset for 3D teeth segmentation.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_points: int = 2048,
        augment: bool = True,
        normalize: bool = True,
        use_normals: bool = True,
        use_colors: bool = False,
        cache_data: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing the data
            split: Dataset split ('train', 'val', 'test')
            num_points: Number of points to sample from each mesh
            augment: Whether to apply data augmentation
            normalize: Whether to normalize point coordinates
            use_normals: Whether to include surface normals as features
            use_colors: Whether to include vertex colors as features
            cache_data: Whether to cache loaded data in memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.augment = augment and (split == 'train')
        self.normalize = normalize
        self.use_normals = use_normals
        self.use_colors = use_colors
        self.cache_data = cache_data
        
        # Load file lists
        self.mesh_files = []
        self.label_files = []
        self._load_file_lists()
        
        # Cache for loaded data
        self.data_cache = {} if cache_data else None
        
        # Feature dimension calculation
        self.feature_dim = 3  # XYZ coordinates
        if use_normals:
            self.feature_dim += 3  # Normal vectors
        if use_colors:
            self.feature_dim += 3  # RGB colors
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f'Loaded {len(self.mesh_files)} samples for {split} split')
    
    def _load_file_lists(self):
        """Load lists of mesh and label files."""
        # Look for split files first
        split_file = self.data_dir / f'{self.split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                file_ids = [line.strip() for line in f if line.strip()]
        else:
            # Fall back to finding all files
            mesh_pattern = "*.obj"
            all_mesh_files = list(self.data_dir.glob(f"**/{mesh_pattern}"))
            
            # Create split based on filename
            random.seed(42)  # For reproducible splits
            random.shuffle(all_mesh_files)
            
            total_files = len(all_mesh_files)
            if self.split == 'train':
                file_ids = [f.stem for f in all_mesh_files[:int(0.7 * total_files)]]
            elif self.split == 'val':
                start_idx = int(0.7 * total_files)
                end_idx = int(0.85 * total_files)
                file_ids = [f.stem for f in all_mesh_files[start_idx:end_idx]]
            else:  # test
                file_ids = [f.stem for f in all_mesh_files[int(0.85 * total_files):]]
        
        # Find corresponding mesh and label files
        for file_id in file_ids:
            mesh_file = self._find_file(file_id, ['.obj'])
            label_file = self._find_file(file_id, ['.json'])
            
            if mesh_file and label_file:
                self.mesh_files.append(mesh_file)
                self.label_files.append(label_file)
    
    def _find_file(self, file_id: str, extensions: List[str]) -> Optional[Path]:
        """Find file with given ID and extensions."""
        for ext in extensions:
            # Try exact match
            file_path = self.data_dir / f"{file_id}{ext}"
            if file_path.exists():
                return file_path
            
            # Try recursive search
            matches = list(self.data_dir.glob(f"**/{file_id}{ext}"))
            if matches:
                return matches[0]
        
        return None
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.mesh_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing points, labels, and metadata
        """
        # Check cache first
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
        
        mesh_file = self.mesh_files[idx]
        label_file = self.label_files[idx]
        
        try:
            # Load mesh and labels
            points, features = self._load_mesh(mesh_file)
            seg_labels, inst_labels, metadata = self._load_labels(label_file)
            
            # Sample points
            sampled_points, sampled_features, sampled_seg_labels, sampled_inst_labels = self._sample_points(
                points, features, seg_labels, inst_labels
            )
            
            # Apply augmentations
            if self.augment:
                sampled_points, sampled_features = self._augment_data(sampled_points, sampled_features)
            
            # Normalize
            if self.normalize:
                sampled_points = self._normalize_points(sampled_points)
            
            # Combine points and features
            if sampled_features is not None:
                point_features = np.concatenate([sampled_points, sampled_features], axis=1)
            else:
                point_features = sampled_points
            
            # Create sample dictionary
            sample = {
                'points': torch.from_numpy(point_features.T).float(),  # [feature_dim, num_points]
                'seg_labels': torch.from_numpy(sampled_seg_labels).long(),
                'inst_labels': torch.from_numpy(sampled_inst_labels).long(),
                'mesh_file': str(mesh_file),
                'jaw_type': metadata.get('jaw', 'unknown'),
                'patient_id': metadata.get('id_patient', 'unknown')
            }
            
            # Cache if enabled
            if self.cache_data:
                self.data_cache[idx] = sample
            
            return sample
            
        except Exception as e:
            self.logger.error(f'Error loading sample {idx}: {e}')
            # Return dummy data to avoid breaking the training loop
            return self._get_dummy_sample()
    
    def _load_mesh(self, mesh_file: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load mesh from file.
        
        Args:
            mesh_file: Path to mesh file
            
        Returns:
            points: Vertex coordinates [N, 3]
            features: Additional features [N, F] or None
        """
        mesh = trimesh.load(str(mesh_file), process=False)
        points = mesh.vertices.astype(np.float32)
        
        features = []
        
        # Add normals if requested
        if self.use_normals:
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                normals = mesh.vertex_normals.astype(np.float32)
            else:
                # Compute approximate normals
                normals = self._compute_vertex_normals(mesh)
            features.append(normals)
        
        # Add colors if requested
        if self.use_colors:
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            else:
                # Default white color
                colors = np.ones((len(points), 3), dtype=np.float32)
            features.append(colors)
        
        # Combine features
        if features:
            features = np.concatenate(features, axis=1)
        else:
            features = None
        
        return points, features
    
    def _compute_vertex_normals(self, mesh) -> np.ndarray:
        """Compute vertex normals from mesh faces."""
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Initialize vertex normals
        vertex_normals = np.zeros_like(vertices)
        
        # Compute face normals and accumulate to vertices
        for face in faces:
            v0, v1, v2 = vertices[face]
            face_normal = np.cross(v1 - v0, v2 - v0)
            face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)
            
            vertex_normals[face] += face_normal
        
        # Normalize vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / (norms + 1e-8)
        
        return vertex_normals.astype(np.float32)
    
    def _load_labels(self, label_file: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load labels from JSON file.
        
        Args:
            label_file: Path to label file
            
        Returns:
            seg_labels: Segmentation labels [N]
            inst_labels: Instance labels [N]
            metadata: Additional metadata
        """
        with open(label_file, 'r') as f:
            data = json.load(f)
        
        seg_labels = np.array(data['labels'], dtype=np.int64)
        inst_labels = np.array(data['instances'], dtype=np.int64)
        
        metadata = {
            'jaw': data.get('jaw', 'unknown'),
            'id_patient': data.get('id_patient', 'unknown')
        }
        
        return seg_labels, inst_labels, metadata
    
    def _sample_points(
        self,
        points: np.ndarray,
        features: Optional[np.ndarray],
        seg_labels: np.ndarray,
        inst_labels: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Sample fixed number of points from the mesh.
        
        Args:
            points: Original points [N, 3]
            features: Original features [N, F] or None
            seg_labels: Segmentation labels [N]
            inst_labels: Instance labels [N]
            
        Returns:
            Sampled data
        """
        num_vertices = len(points)
        
        if num_vertices <= self.num_points:
            # Pad with random points if not enough vertices
            indices = np.arange(num_vertices)
            pad_size = self.num_points - num_vertices
            pad_indices = np.random.choice(num_vertices, pad_size, replace=True)
            indices = np.concatenate([indices, pad_indices])
        else:
            # Random sampling
            indices = np.random.choice(num_vertices, self.num_points, replace=False)
        
        sampled_points = points[indices]
        sampled_features = features[indices] if features is not None else None
        sampled_seg_labels = seg_labels[indices]
        sampled_inst_labels = inst_labels[indices]
        
        return sampled_points, sampled_features, sampled_seg_labels, sampled_inst_labels
    
    def _augment_data(
        self,
        points: np.ndarray,
        features: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply data augmentation to points and features.
        
        Args:
            points: Point coordinates [N, 3]
            features: Point features [N, F] or None
            
        Returns:
            Augmented points and features
        """
        # Random rotation around Z-axis (jaw rotation)
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T
        
        # Apply rotation to normals if present
        if features is not None and self.use_normals:
            start_idx = 3 if self.use_normals else 0
            if start_idx < features.shape[1]:
                normals = features[:, start_idx:start_idx+3]
                normals = normals @ rotation_matrix.T
                features[:, start_idx:start_idx+3] = normals
        
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        points *= scale
        
        # Random translation
        translation = np.random.uniform(-0.1, 0.1, 3)
        points += translation
        
        # Add noise
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        
        # Random jittering of features
        if features is not None:
            feature_noise = np.random.normal(0, 0.02, features.shape)
            features += feature_noise
        
        return points, features
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize point coordinates.
        
        Args:
            points: Point coordinates [N, 3]
            
        Returns:
            Normalized points
        """
        # Center points
        centroid = points.mean(axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        return points
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return dummy sample for error handling."""
        dummy_points = torch.randn(self.feature_dim, self.num_points)
        dummy_seg_labels = torch.zeros(self.num_points, dtype=torch.long)
        dummy_inst_labels = torch.zeros(self.num_points, dtype=torch.long)
        
        return {
            'points': dummy_points,
            'seg_labels': dummy_seg_labels,
            'inst_labels': dummy_inst_labels,
            'mesh_file': 'dummy',
            'jaw_type': 'unknown',
            'patient_id': 'unknown'
        }


class TeethSegmentationDataModule:
    """
    Data module for handling train/val/test data loaders.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        num_points: int = 2048,
        augment: bool = True,
        pin_memory: bool = True,
        **dataset_kwargs
    ):
        """
        Initialize data module.
        
        Args:
            data_dir: Root directory containing the data
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            num_points: Number of points to sample
            augment: Whether to apply augmentation
            pin_memory: Whether to pin memory for faster GPU transfer
            **dataset_kwargs: Additional arguments for dataset
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_points = num_points
        self.augment = augment
        self.pin_memory = pin_memory
        self.dataset_kwargs = dataset_kwargs
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup train/val/test datasets."""
        self.train_dataset = TeethSegmentationDataset(
            data_dir=self.data_dir,
            split='train',
            num_points=self.num_points,
            augment=self.augment,
            **self.dataset_kwargs
        )
        
        self.val_dataset = TeethSegmentationDataset(
            data_dir=self.data_dir,
            split='val',
            num_points=self.num_points,
            augment=False,  # No augmentation for validation
            **self.dataset_kwargs
        )
        
        self.test_dataset = TeethSegmentationDataset(
            data_dir=self.data_dir,
            split='test',
            num_points=self.num_points,
            augment=False,  # No augmentation for testing
            **self.dataset_kwargs
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched data dictionary
    """
    # Stack tensors
    points = torch.stack([sample['points'] for sample in batch])
    seg_labels = torch.stack([sample['seg_labels'] for sample in batch])
    inst_labels = torch.stack([sample['inst_labels'] for sample in batch])
    
    # Collect metadata
    mesh_files = [sample['mesh_file'] for sample in batch]
    jaw_types = [sample['jaw_type'] for sample in batch]
    patient_ids = [sample['patient_id'] for sample in batch]
    
    return {
        'points': points,
        'seg_labels': seg_labels,
        'inst_labels': inst_labels,
        'mesh_files': mesh_files,
        'jaw_types': jaw_types,
        'patient_ids': patient_ids
    }


def test_dataset():
    """Test function for the dataset."""
    # Create dummy data directory structure
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / 'test_data'
        data_dir.mkdir()
        
        # Create dummy mesh and label files
        for i in range(10):
            # Create dummy mesh
            vertices = np.random.rand(1000, 3).astype(np.float32)
            faces = np.random.randint(0, 1000, (500, 3))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            mesh_file = data_dir / f'scan_{i:03d}.obj'
            mesh.export(str(mesh_file))
            
            # Create dummy labels
            labels_data = {
                'id_patient': f'patient_{i:03d}',
                'jaw': 'upper' if i % 2 == 0 else 'lower',
                'labels': np.random.randint(11, 48, 1000).tolist(),
                'instances': np.random.randint(0, 16, 1000).tolist()
            }
            
            label_file = data_dir / f'scan_{i:03d}.json'
            with open(label_file, 'w') as f:
                json.dump(labels_data, f)
        
        # Test dataset
        print("Testing TeethSegmentationDataset...")
        dataset = TeethSegmentationDataset(
            data_dir=str(data_dir),
            split='train',
            num_points=512,
            augment=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Feature dimension: {dataset.feature_dim}")
        
        # Test single sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Points shape: {sample['points'].shape}")
        print(f"Seg labels shape: {sample['seg_labels'].shape}")
        print(f"Inst labels shape: {sample['inst_labels'].shape}")
        
        # Test data loader
        print("\nTesting DataLoader...")
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        for batch in data_loader:
            print(f"Batch points shape: {batch['points'].shape}")
            print(f"Batch seg labels shape: {batch['seg_labels'].shape}")
            break
        
        print("✓ Dataset test completed successfully!")


if __name__ == "__main__":
    test_dataset()