#!/usr/bin/env python3
"""
3D Teeth Segmentation Visualization Tools

This module provides visualization utilities for 3D teeth segmentation results,
including mesh rendering, metric plotting, and comparative analysis.

Author: Enhanced for academic research
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import trimesh
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path
import pandas as pd


class TeethVisualization:
    """Visualization tools for 3D teeth segmentation results."""
    
    def __init__(self):
        """Initialize visualization parameters."""
        self.fdi_colors = self._get_fdi_color_map()
        plt.style.use('seaborn-v0_8')
        
    def _get_fdi_color_map(self):
        """Get FDI tooth numbering system color mapping."""
        return {
            # Upper jaw (maxilla)
            11: '#FF0000', 12: '#FF3300', 13: '#FF6600', 14: '#FF9900', 15: '#FFCC00', 16: '#FFFF00', 17: '#CCFF00', 18: '#99FF00',
            21: '#66FF00', 22: '#33FF00', 23: '#00FF00', 24: '#00FF33', 25: '#00FF66', 26: '#00FF99', 27: '#00FFCC', 28: '#00FFFF',
            # Lower jaw (mandible)  
            31: '#00CCFF', 32: '#0099FF', 33: '#0066FF', 34: '#0033FF', 35: '#0000FF', 36: '#3300FF', 37: '#6600FF', 38: '#9900FF',
            41: '#CC00FF', 42: '#FF00CC', 43: '#FF0099', 44: '#FF0066', 45: '#FF0033', 46: '#FF0000', 47: '#CC0000', 48: '#990000',
            0: '#808080'  # Gingiva (gray)
        }
    
    def visualize_3d_mesh(self, mesh_path, labels_path, title="3D Teeth Segmentation"):
        """
        Visualize 3D mesh with tooth labels using Plotly.
        
        Args:
            mesh_path (str): Path to .obj mesh file
            labels_path (str): Path to labels JSON file
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive 3D plot
        """
        # Load mesh and labels
        mesh = trimesh.load(mesh_path, process=False)
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        labels = np.array(labels_data['labels'])
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create color array based on tooth labels
        colors = np.array([self.fdi_colors.get(label, '#808080') for label in labels])
        
        # Create 3D mesh plot
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1], 
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=colors,
                opacity=0.8,
                name="Teeth"
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        Plot comparison of evaluation metrics.
        
        Args:
            metrics_dict (dict): Dictionary containing metric values
            save_path (str): Optional path to save the plot
        """
        metrics = ['TSA', 'TLA', 'TIR', 'precision', 'recall', 'iou', 'dice']
        values = [metrics_dict.get(m, 0) for m in metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(metrics, values, color=plt.cm.viridis(np.linspace(0, 1, len(metrics))))
        ax1.set_ylabel('Score')
        ax1.set_title('Evaluation Metrics Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Radar plot
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        values_radar = values + [values[0]]  # Complete the circle
        angles_radar = np.concatenate((angles, [angles[0]]))
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles_radar, values_radar, 'o-', linewidth=2, color='blue')
        ax2.fill(angles_radar, values_radar, alpha=0.25, color='blue')
        ax2.set_xticks(angles)
        ax2.set_xticklabels(metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('Metrics Radar Chart')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per_tooth_analysis(self, per_tooth_metrics, jaw_type, save_path=None):
        """
        Plot per-tooth analysis showing IoU and Dice coefficients.
        
        Args:
            per_tooth_metrics (dict): Per-tooth metrics dictionary
            jaw_type (str): 'upper' or 'lower' jaw
            save_path (str): Optional path to save the plot
        """
        if not per_tooth_metrics:
            print("No per-tooth metrics available")
            return None
            
        tooth_labels = list(per_tooth_metrics.keys())
        iou_scores = [per_tooth_metrics[tooth]['iou'] for tooth in tooth_labels]
        dice_scores = [per_tooth_metrics[tooth]['dice'] for tooth in tooth_labels]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # IoU scores
        x_pos = np.arange(len(tooth_labels))
        bars1 = ax1.bar(x_pos, iou_scores, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Tooth Label (FDI)')
        ax1.set_ylabel('IoU Score')
        ax1.set_title(f'Per-Tooth IoU Scores - {jaw_type.title()} Jaw')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(tooth_labels, rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars1, iou_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Dice scores
        bars2 = ax2.bar(x_pos, dice_scores, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Tooth Label (FDI)')
        ax2.set_ylabel('Dice Score')
        ax2.set_title(f'Per-Tooth Dice Scores - {jaw_type.title()} Jaw')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(tooth_labels, rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars2, dice_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_confusion_matrix_heatmap(self, gt_labels, pred_labels, save_path=None):
        """
        Create confusion matrix heatmap for tooth classification.
        
        Args:
            gt_labels (array): Ground truth labels
            pred_labels (array): Predicted labels
            save_path (str): Optional path to save the plot
        """
        from sklearn.metrics import confusion_matrix
        
        # Get unique labels
        unique_labels = sorted(list(set(gt_labels) | set(pred_labels)))
        
        # Create confusion matrix
        cm = confusion_matrix(gt_labels, pred_labels, labels=unique_labels)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=unique_labels,
                   yticklabels=unique_labels)
        
        plt.title('Tooth Classification Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_training_curves(self, train_losses, val_losses, metrics_history=None, save_path=None):
        """
        Plot training and validation curves.
        
        Args:
            train_losses (list): Training loss values
            val_losses (list): Validation loss values  
            metrics_history (dict): Optional metrics history
            save_path (str): Optional path to save the plot
        """
        epochs = range(1, len(train_losses) + 1)
        
        if metrics_history:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        if metrics_history:
            # Accuracy/F1 curves
            if 'train_f1' in metrics_history:
                ax2.plot(epochs, metrics_history['train_f1'], 'b-', label='Training F1')
                ax2.plot(epochs, metrics_history['val_f1'], 'r-', label='Validation F1')
                ax2.set_title('F1 Score')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('F1 Score')
                ax2.legend()
                ax2.grid(True)
            
            # IoU curves
            if 'train_iou' in metrics_history:
                ax3.plot(epochs, metrics_history['train_iou'], 'b-', label='Training IoU')
                ax3.plot(epochs, metrics_history['val_iou'], 'r-', label='Validation IoU')
                ax3.set_title('IoU Score')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('IoU')
                ax3.legend()
                ax3.grid(True)
            
            # Learning rate
            if 'learning_rate' in metrics_history:
                ax4.plot(epochs, metrics_history['learning_rate'], 'g-')
                ax4.set_title('Learning Rate')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Learning Rate')
                ax4.set_yscale('log')
                ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_sample_visualization():
    """Create sample visualizations for demonstration."""
    viz = TeethVisualization()
    
    # Sample metrics data
    sample_metrics = {
        'TSA': 0.92,
        'TLA': 0.88,
        'TIR': 0.85,
        'precision': 0.91,
        'recall': 0.89,
        'iou': 0.84,
        'dice': 0.87
    }
    
    # Create metrics comparison plot
    fig1 = viz.plot_metrics_comparison(sample_metrics, 'sample_metrics.png')
    
    # Sample per-tooth metrics
    sample_per_tooth = {
        11: {'iou': 0.89, 'dice': 0.92},
        12: {'iou': 0.91, 'dice': 0.94},
        13: {'iou': 0.87, 'dice': 0.90},
        14: {'iou': 0.85, 'dice': 0.88},
        15: {'iou': 0.83, 'dice': 0.86},
        16: {'iou': 0.86, 'dice': 0.89}
    }
    
    # Create per-tooth analysis
    fig2 = viz.plot_per_tooth_analysis(sample_per_tooth, 'upper', 'sample_per_tooth.png')
    
    plt.show()


if __name__ == "__main__":
    create_sample_visualization()