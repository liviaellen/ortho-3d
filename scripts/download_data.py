#!/usr/bin/env python3
"""
Download and setup real dental data from OSF repository
3DTeethSeg22 Challenge Dataset

This script downloads a subset of the official dataset for training.
"""

import os
import urllib.request
import zipfile
import json
import shutil
from pathlib import Path
import sys

def create_data_structure():
    """Create the required data directory structure."""
    print("📁 Creating data directory structure...")
    
    directories = [
        "data",
        "data/raw",
        "data/scans",
        "data/labels", 
        "data/train",
        "data/val",
        "data/test",
        "checkpoints",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {dir_path}")

def download_sample_data():
    """
    Download a small subset of real data for training.
    
    Note: For the full dataset, manually download from:
    https://osf.io/xctdy/
    """
    print("🌐 Setting up data download...")
    
    # Create instructions file for manual download
    instructions = """
# 📥 Real Dental Data Download Instructions

## Option 1: Download Full Dataset (Recommended)
1. Visit: https://osf.io/xctdy/
2. Download all 6 parts (or just Part 1 for testing)
3. Extract to: ./data/raw/
4. Run: python scripts/organize_data.py

## Option 2: Quick Test with Sample Data
The system can work with any .obj + .json files:
1. Place .obj files in: ./data/scans/
2. Place corresponding .json labels in: ./data/labels/
3. Run training script

## Dataset Structure Expected:
```
data/
├── scans/
│   ├── patient_001_upper.obj
│   ├── patient_001_lower.obj
│   └── ...
└── labels/
    ├── patient_001_upper.json
    ├── patient_001_lower.json
    └── ...
```

## Sample JSON Format:
```json
{
    "id_patient": "001", 
    "jaw": "upper",
    "labels": [0, 0, 11, 12, 13, ...],
    "instances": [0, 0, 1, 2, 3, ...]
}
```

## Next Steps After Download:
1. python scripts/organize_data.py
2. python scripts/train_real_data.py
"""
    
    with open("data/DOWNLOAD_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("  ✅ Created download instructions: data/DOWNLOAD_INSTRUCTIONS.md")
    print("  📋 Follow the instructions to get real dental data")

def create_dummy_sample():
    """Create a small dummy sample for immediate testing."""
    print("🎲 Creating dummy sample data...")
    
    import numpy as np
    import trimesh
    
    # Create sample dental arch
    angles = np.linspace(-np.pi/3, np.pi/3, 8)
    radius = 3.0
    
    vertices = []
    faces = []
    labels = []
    instances = []
    
    vertex_count = 0
    
    for i, angle in enumerate(angles):
        x_center = radius * np.cos(angle)
        z_center = radius * np.sin(angle)
        
        # Simple tooth shape
        tooth_vertices = []
        for layer in range(5):
            y = (layer/4) * 0.8
            for corner in range(8):
                corner_angle = (corner/8) * 2 * np.pi
                x = x_center + 0.3 * np.cos(corner_angle)
                z = z_center + 0.3 * np.sin(corner_angle)
                tooth_vertices.append([x, y, z])
        
        # Create faces
        tooth_faces = []
        for layer in range(4):
            for corner in range(8):
                v1 = vertex_count + layer * 8 + corner
                v2 = vertex_count + layer * 8 + ((corner + 1) % 8)
                v3 = vertex_count + (layer + 1) * 8 + corner
                v4 = vertex_count + (layer + 1) * 8 + ((corner + 1) % 8)
                tooth_faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        vertices.extend(tooth_vertices)
        faces.extend(tooth_faces)
        
        # FDI labels
        fdi_label = 11 + i if i < 4 else 21 + (i - 4)
        labels.extend([fdi_label] * len(tooth_vertices))
        instances.extend([i + 1] * len(tooth_vertices))
        vertex_count += len(tooth_vertices)
    
    # Add gingiva vertices
    gingiva_count = 200
    gingiva_vertices = np.random.rand(gingiva_count, 3) * 6 - 3
    gingiva_vertices[:, 1] = -0.2  # Below teeth
    
    vertices.extend(gingiva_vertices.tolist())
    labels.extend([0] * gingiva_count)  # Gingiva label
    instances.extend([0] * gingiva_count)  # Gingiva instance
    
    # Create mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Add noise for realism
    noise = np.random.normal(0, 0.02, vertices.shape)
    vertices += noise
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces[:len(faces)//2])  # Reduce faces
    
    # Save mesh
    mesh_path = "data/scans/sample_upper_jaw.obj"
    mesh.export(mesh_path)
    
    # Save labels
    label_data = {
        "id_patient": "SAMPLE001",
        "jaw": "upper", 
        "labels": labels,
        "instances": instances
    }
    
    label_path = "data/labels/sample_upper_jaw.json"
    with open(label_path, "w") as f:
        json.dump(label_data, f, indent=2)
    
    print(f"  ✅ Created sample mesh: {mesh_path}")
    print(f"  ✅ Created sample labels: {label_path}")
    print(f"  📊 Sample stats: {len(vertices)} vertices, {len(np.unique(instances))} teeth")

def main():
    """Main function to set up data."""
    print("🦷 3D Teeth Segmentation - Data Setup")
    print("=" * 50)
    
    # Create directory structure
    create_data_structure()
    print()
    
    # Download/setup instructions
    download_sample_data()
    print()
    
    # Create dummy sample
    create_dummy_sample()
    print()
    
    print("✅ Data setup completed!")
    print("\n📋 Next steps:")
    print("1. Follow instructions in: data/DOWNLOAD_INSTRUCTIONS.md")
    print("2. Or use dummy data with: python scripts/train_real_data.py")
    print("3. For full dataset: Download from https://osf.io/xctdy/")

if __name__ == "__main__":
    main()