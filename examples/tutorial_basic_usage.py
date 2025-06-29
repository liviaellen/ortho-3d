#!/usr/bin/env python3
"""
Basic Usage Tutorial for 3D Teeth Segmentation Challenge

This tutorial demonstrates how to:
1. Load and visualize 3D dental scans
2. Run evaluation metrics
3. Create visualizations
4. Analyze results

Author: Enhanced for academic research
"""

import sys
import os
import json
import numpy as np
import trimesh
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.evaluation import calculate_metrics
from visualization.visualize_results import TeethVisualization


def load_sample_data():
    """
    Load sample data for demonstration.
    Replace with your actual data loading logic.
    """
    print("Loading sample dental scan data...")
    
    # This is a mock example - replace with actual data loading
    sample_gt_data = {
        'mesh_vertices': np.random.rand(10000, 3) * 50,  # 10k vertices
        'instances': np.random.randint(0, 17, 10000),    # Random tooth instances
        'labels': np.random.randint(11, 48, 10000)       # FDI labels
    }
    
    sample_pred_data = {
        'instances': np.random.randint(0, 17, 10000),    # Predicted instances  
        'labels': np.random.randint(11, 48, 10000)       # Predicted FDI labels
    }
    
    return sample_gt_data, sample_pred_data


def tutorial_evaluation_metrics():
    """
    Tutorial: How to calculate evaluation metrics
    """
    print("\n=== TUTORIAL: Evaluation Metrics ===")
    
    # Load sample data
    gt_data, pred_data = load_sample_data()
    
    # Calculate all metrics
    print("Calculating evaluation metrics...")
    jaw_TLA, jaw_TSA, jaw_TIR, precision, recall, iou, dice, per_tooth_metrics = calculate_metrics(
        gt_data, pred_data
    )
    
    # Display results
    print(f"\nResults:")
    print(f"Teeth Localization Accuracy (TLA): {jaw_TLA:.4f}")
    print(f"Teeth Segmentation Accuracy (TSA): {jaw_TSA:.4f}")
    print(f"Teeth Identification Rate (TIR): {jaw_TIR:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"IoU (Intersection over Union): {iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")
    
    # Calculate overall score (as used in challenge)
    exp_neg_tla = np.exp(-jaw_TLA)
    overall_score = (jaw_TSA + exp_neg_tla + jaw_TIR) / 3
    print(f"Overall Challenge Score: {overall_score:.4f}")
    
    # Display per-tooth metrics (if available)
    if per_tooth_metrics:
        print(f"\nPer-tooth metrics available for {len(per_tooth_metrics)} teeth")
        for tooth_label, metrics in list(per_tooth_metrics.items())[:3]:  # Show first 3
            print(f"Tooth {tooth_label}: IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}")
    
    return jaw_TLA, jaw_TSA, jaw_TIR, precision, recall, iou, dice, per_tooth_metrics


def tutorial_visualization():
    """
    Tutorial: How to create visualizations
    """
    print("\n=== TUTORIAL: Data Visualization ===")
    
    # Load sample data
    gt_data, pred_data = load_sample_data()
    
    # Calculate metrics
    jaw_TLA, jaw_TSA, jaw_TIR, precision, recall, iou, dice, per_tooth_metrics = calculate_metrics(
        gt_data, pred_data
    )
    
    # Initialize visualization
    viz = TeethVisualization()
    
    # 1. Metrics comparison plot
    print("Creating metrics comparison plot...")
    metrics_dict = {
        'TSA': jaw_TSA,
        'TLA': np.exp(-jaw_TLA),  # Use exp(-TLA) for visualization
        'TIR': jaw_TIR,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'dice': dice
    }
    
    fig1 = viz.plot_metrics_comparison(metrics_dict, save_path='metrics_comparison.png')
    print("✓ Metrics comparison plot saved as 'metrics_comparison.png'")
    
    # 2. Per-tooth analysis (if available)
    if per_tooth_metrics:
        print("Creating per-tooth analysis plot...")
        fig2 = viz.plot_per_tooth_analysis(per_tooth_metrics, 'mixed', save_path='per_tooth_analysis.png')
        print("✓ Per-tooth analysis plot saved as 'per_tooth_analysis.png'")
    
    # 3. Training curves example
    print("Creating sample training curves...")
    # Generate sample training data
    epochs = 50
    train_losses = np.exp(-np.linspace(2, 0.5, epochs)) + np.random.normal(0, 0.02, epochs)
    val_losses = np.exp(-np.linspace(2, 0.8, epochs)) + np.random.normal(0, 0.03, epochs)
    
    metrics_history = {
        'train_f1': np.tanh(np.linspace(0, 2, epochs)) + np.random.normal(0, 0.02, epochs),
        'val_f1': np.tanh(np.linspace(0, 1.8, epochs)) + np.random.normal(0, 0.03, epochs),
        'train_iou': np.tanh(np.linspace(0, 1.9, epochs)) + np.random.normal(0, 0.02, epochs),
        'val_iou': np.tanh(np.linspace(0, 1.7, epochs)) + np.random.normal(0, 0.03, epochs),
        'learning_rate': np.exp(-np.linspace(0, 3, epochs)) * 0.001
    }
    
    fig3 = viz.plot_training_curves(train_losses, val_losses, metrics_history, save_path='training_curves.png')
    print("✓ Training curves plot saved as 'training_curves.png'")
    
    return fig1, fig2 if per_tooth_metrics else None, fig3


def tutorial_data_loading():
    """
    Tutorial: How to load real dental scan data
    """
    print("\n=== TUTORIAL: Data Loading ===")
    
    # Example of loading .obj file and JSON labels
    print("Example code for loading real dental scan data:")
    
    code_example = '''
# Load 3D mesh (.obj file)
import trimesh
mesh = trimesh.load('path/to/dental_scan.obj', process=False)
vertices = mesh.vertices  # Nx3 array of vertex coordinates
faces = mesh.faces        # Mx3 array of face indices

# Load ground truth labels (JSON file)
import json
with open('path/to/labels.json', 'r') as f:
    labels_data = json.load(f)

gt_labels = labels_data['labels']        # Per-vertex tooth labels (FDI system)
gt_instances = labels_data['instances']  # Per-vertex instance IDs
jaw_type = labels_data['jaw']           # 'upper' or 'lower'
patient_id = labels_data['id_patient']  # Patient identifier

# Prepare data for evaluation
gt_data = {
    'mesh_vertices': vertices,
    'instances': gt_instances,
    'labels': gt_labels
}

# Load predictions (same format)
with open('path/to/predictions.json', 'r') as f:
    pred_data = json.load(f)
    '''
    
    print(code_example)
    
    # Explain FDI numbering system
    print("\nFDI Tooth Numbering System:")
    print("- Upper right: 11-18 (central incisor to wisdom tooth)")
    print("- Upper left: 21-28")  
    print("- Lower left: 31-38")
    print("- Lower right: 41-48")
    print("- Gingiva (gums): 0")


def tutorial_batch_evaluation():
    """
    Tutorial: How to evaluate multiple files in batch
    """
    print("\n=== TUTORIAL: Batch Evaluation ===")
    
    batch_code = '''
import glob
from pathlib import Path

def evaluate_batch(gt_dir, pred_dir, output_file):
    """
    Evaluate multiple dental scans in batch.
    
    Args:
        gt_dir: Directory containing ground truth JSON files
        pred_dir: Directory containing prediction JSON files
        output_file: File to save batch results
    """
    results = []
    
    # Find all ground truth files
    gt_files = glob.glob(f"{gt_dir}/*.json")
    
    for gt_file in gt_files:
        filename = Path(gt_file).stem
        pred_file = f"{pred_dir}/{filename}.json"
        
        if not os.path.exists(pred_file):
            print(f"Warning: Prediction file missing for {filename}")
            continue
        
        # Load data
        with open(gt_file) as f:
            gt_data = json.load(f)
        with open(pred_file) as f:
            pred_data = json.load(f)
        
        # Calculate metrics
        jaw_TLA, jaw_TSA, jaw_TIR, precision, recall, iou, dice, per_tooth = calculate_metrics(gt_data, pred_data)
        
        # Store results
        results.append({
            'filename': filename,
            'TLA': jaw_TLA,
            'TSA': jaw_TSA,
            'TIR': jaw_TIR,
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'dice': dice,
            'exp_neg_tla': np.exp(-jaw_TLA),
            'overall_score': (jaw_TSA + np.exp(-jaw_TLA) + jaw_TIR) / 3
        })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    if results:
        mean_score = np.mean([r['overall_score'] for r in results])
        print(f"Batch evaluation complete. Mean score: {mean_score:.4f}")
    
    return results

# Usage example:
# results = evaluate_batch('data/gt/', 'data/predictions/', 'batch_results.json')
    '''
    
    print(batch_code)


def main():
    """
    Main tutorial function
    """
    print("🦷 3D Teeth Segmentation Challenge - Tutorial")
    print("=" * 50)
    
    # Run all tutorials
    try:
        # 1. Evaluation metrics
        tutorial_evaluation_metrics()
        
        # 2. Visualization
        tutorial_visualization()
        
        # 3. Data loading
        tutorial_data_loading()
        
        # 4. Batch evaluation
        tutorial_batch_evaluation()
        
        print("\n✅ Tutorial completed successfully!")
        print("\nNext steps:")
        print("1. Replace sample data with your actual dental scans")
        print("2. Implement your segmentation algorithm")
        print("3. Use the evaluation functions to assess performance")
        print("4. Create visualizations to analyze results")
        
    except Exception as e:
        print(f"\n❌ Tutorial failed with error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install trimesh numpy scikit-learn scipy matplotlib plotly seaborn pandas")


if __name__ == "__main__":
    main()