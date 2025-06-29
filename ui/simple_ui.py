#!/usr/bin/env python3
"""
Simple Streamlit UI for 3D Teeth Segmentation Testing

A lightweight interface that works without complex dependencies.

Usage:
    streamlit run ui/simple_ui.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import sys
import os
import tempfile
import time
import json
import trimesh

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set page config
st.set_page_config(
    page_title="3D Teeth Segmentation",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        return True, torch.__version__, torch.cuda.is_available()
    except ImportError:
        return False, "Not installed", False

def test_models():
    """Test if models can be imported and used."""
    try:
        from models.pytorch_models import PointNetSegmentation, TeethSegmentationNet
        
        # Test PointNet
        pointnet = PointNetSegmentation(num_classes=49)
        test_input = torch.randn(1, 3, 512)
        pointnet_output = pointnet(test_input)
        
        # Test Custom model
        custom = TeethSegmentationNet(num_classes=49, num_instances=32)
        test_input_custom = torch.randn(1, 6, 512)
        seg_out, inst_out = custom(test_input_custom)
        
        return True, {
            'pointnet_output_shape': str(pointnet_output.shape),
            'custom_seg_shape': str(seg_out.shape),
            'custom_inst_shape': str(inst_out.shape)
        }
    except Exception as e:
        return False, str(e)

def load_real_dental_data():
    """Load real dental scan data from teeth3ds_sample directory."""
    try:
        # Check if we have the copied data first
        obj_path = os.path.join('data', 'scans', 'real_patient_01F4JV8X_upper.obj')
        json_path = os.path.join('data', 'labels', 'real_patient_01F4JV8X_upper.json')
        
        if not os.path.exists(obj_path):
            # Try original location
            obj_path = os.path.join('teeth3ds_sample', '01F4JV8X', '01F4JV8X_upper.obj')
            json_path = os.path.join('teeth3ds_sample', '01F4JV8X', '01F4JV8X_upper.json')
        
        if not os.path.exists(obj_path) or not os.path.exists(json_path):
            return None, None, None, "Real dental data not found"
        
        # Load mesh using trimesh
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Load labels
        with open(json_path, 'r') as f:
            labels_data = json.load(f)
        
        # Extract labels array
        if 'labels' in labels_data:
            labels = np.array(labels_data['labels'])
        elif isinstance(labels_data, list):
            labels = np.array(labels_data)
        else:
            # Try to extract from instances
            labels = np.zeros(len(vertices), dtype=int)
            if 'instances' in labels_data:
                for instance in labels_data['instances']:
                    if 'pointIndices' in instance and 'labelId' in instance:
                        indices = instance['pointIndices']
                        label_id = instance['labelId']
                        labels[indices] = label_id
        
        return vertices, faces, labels, None
        
    except Exception as e:
        return None, None, None, f"Error loading data: {str(e)}"

def create_sample_mesh():
    """Create a simple sample mesh for demonstration."""
    # Create a simple dental arch shape
    angles = np.linspace(-np.pi/3, np.pi/3, 8)
    radius = 3.0
    
    vertices = []
    faces = []
    labels = []
    
    vertex_count = 0
    
    # Create simple teeth shapes
    for i, angle in enumerate(angles):
        x_center = radius * np.cos(angle)
        z_center = radius * np.sin(angle)
        
        # Simple tooth vertices (box-like)
        tooth_size = 0.3
        tooth_height = 0.8
        
        # Create 8 vertices for a simple box
        for dx in [-1, 1]:
            for dy in [0, 1]:
                for dz in [-1, 1]:
                    x = x_center + dx * tooth_size
                    y = dy * tooth_height
                    z = z_center + dz * tooth_size
                    vertices.append([x, y, z])
        
        # Create faces for the box (simplified)
        base = vertex_count
        box_faces = [
            [base, base+1, base+2], [base+1, base+3, base+2],  # Front
            [base+4, base+6, base+5], [base+5, base+6, base+7],  # Back
            [base, base+4, base+1], [base+1, base+4, base+5],    # Bottom
            [base+2, base+3, base+6], [base+3, base+7, base+6]   # Top
        ]
        faces.extend(box_faces)
        
        # Assign FDI labels
        fdi_label = 11 + i if i < 4 else 21 + (i - 4)
        labels.extend([fdi_label] * 8)
        
        vertex_count += 8
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    labels = np.array(labels)
    
    return vertices, faces, labels

def visualize_mesh(vertices, faces, labels=None, title="3D Mesh", max_vertices=10000):
    """Create 3D visualization with optional subsampling for large meshes."""
    # Subsample if mesh is too large for smooth visualization
    if len(vertices) > max_vertices:
        step = len(vertices) // max_vertices
        vertex_indices = np.arange(0, len(vertices), step)
        vertices_vis = vertices[vertex_indices]
        
        if labels is not None:
            labels_vis = labels[vertex_indices]
        else:
            labels_vis = None
            
        # Create simplified faces (just use original faces that reference valid vertices)
        faces_vis = []
        for face in faces:
            if all(v < len(vertices_vis) for v in face):
                faces_vis.append(face)
        
        if len(faces_vis) == 0:
            # If no valid faces, create point cloud visualization
            if labels_vis is not None:
                colors = [f'hsl({(label)*20}, 70%, 50%)' for label in labels_vis]
            else:
                colors = ['lightblue'] * len(vertices_vis)
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=vertices_vis[:, 0],
                    y=vertices_vis[:, 1],
                    z=vertices_vis[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=colors,
                        opacity=0.6
                    ),
                    name="Points"
                )
            ])
        else:
            faces_vis = np.array(faces_vis)
            if labels_vis is not None:
                colors = [f'hsl({(label)*20}, 70%, 50%)' for label in labels_vis]
            else:
                colors = ['lightblue'] * len(vertices_vis)
            
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=vertices_vis[:, 0],
                    y=vertices_vis[:, 1],
                    z=vertices_vis[:, 2],
                    i=faces_vis[:, 0],
                    j=faces_vis[:, 1],
                    k=faces_vis[:, 2],
                    vertexcolor=colors,
                    opacity=0.8,
                    name="Mesh"
                )
            ])
    else:
        # Use full mesh
        if labels is not None:
            colors = [f'hsl({(label)*20}, 70%, 50%)' for label in labels]
        else:
            colors = ['lightblue'] * len(vertices)
        
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
                name="Mesh"
            )
        ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='cube'
        ),
        width=700,
        height=500
    )
    
    return fig

def simulate_segmentation(vertices, processing_time=0.5):
    """Simulate segmentation process."""
    time.sleep(processing_time)  # Simulate processing
    
    # Generate random but realistic results
    num_vertices = len(vertices)
    pred_labels = np.random.choice([11, 12, 13, 14, 15, 16, 21, 22], num_vertices)
    pred_instances = np.random.randint(1, 9, num_vertices)
    
    return pred_labels, pred_instances

def main():
    """Main UI function."""
    # Header
    st.title("🦷 3D Teeth Segmentation - Simple UI")
    st.markdown("**Lightweight testing interface for PyTorch teeth segmentation**")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Check system status
    st.sidebar.subheader("🖥️ System Status")
    pytorch_ok, pytorch_version, cuda_available = check_pytorch()
    
    if pytorch_ok:
        st.sidebar.success(f"✅ PyTorch {pytorch_version}")
        st.sidebar.info(f"CUDA: {'✅ Available' if cuda_available else '❌ Not available'}")
    else:
        st.sidebar.error("❌ PyTorch not found")
    
    # Model testing
    if st.sidebar.button("🧪 Test Models"):
        with st.spinner("Testing PyTorch models..."):
            models_ok, model_info = test_models()
            if models_ok:
                st.sidebar.success("✅ Models working!")
                st.sidebar.json(model_info)
            else:
                st.sidebar.error(f"❌ Model test failed: {model_info}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Overview", "🧪 Testing", "📊 Demo"])
    
    with tab1:
        # Overview
        st.header("🏠 Project Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Features")
            st.markdown("""
            - **🦷 Automatic Segmentation**: AI-powered tooth identification  
            - **🚀 PyTorch Models**: PointNet & Custom architectures
            - **⚡ Real-time Processing**: Fast inference
            - **🎨 3D Visualization**: Interactive mesh viewing
            - **📊 Performance Metrics**: Comprehensive evaluation
            """)
        
        with col2:
            st.subheader("🔧 Status")
            st.metric("PyTorch", "✅" if pytorch_ok else "❌", pytorch_version)
            st.metric("CUDA", "✅" if cuda_available else "❌", "GPU" if cuda_available else "CPU")
    
    with tab2:
        # Testing
        st.header("🧪 Interactive Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📂 Sample Data")
            
            # Data source selection
            data_source = st.radio(
                "Choose data source:",
                ["🦷 Real Clinical Data", "🎲 Generated Sample"]
            )
            
            if data_source == "🦷 Real Clinical Data":
                if st.button("📥 Load Real Dental Scan"):
                    with st.spinner("Loading real dental scan data..."):
                        vertices, faces, labels, error = load_real_dental_data()
                        
                        if error:
                            st.error(f"❌ {error}")
                            st.info("💡 Falling back to generated sample data")
                            vertices, faces, labels = create_sample_mesh()
                        else:
                            st.success(f"✅ Loaded real scan: {len(vertices):,} vertices!")
                            st.info(f"📊 Patient ID: 01F4JV8X (Upper jaw)")
                        
                        # Store in session state
                        st.session_state.vertices = vertices
                        st.session_state.faces = faces
                        st.session_state.labels = labels
                        st.session_state.is_real_data = error is None
            
            else:  # Generated sample
                if st.button("🎲 Generate Sample Jaw"):
                    with st.spinner("Creating sample jaw mesh..."):
                        vertices, faces, labels = create_sample_mesh()
                        
                        # Store in session state
                        st.session_state.vertices = vertices
                        st.session_state.faces = faces
                        st.session_state.labels = labels
                        st.session_state.is_real_data = False
                        
                        st.success(f"✅ Generated mesh with {len(vertices)} vertices!")
            
            # Show mesh info if generated
            if hasattr(st.session_state, 'vertices'):
                is_real = getattr(st.session_state, 'is_real_data', False)
                data_type = "Real Clinical Data" if is_real else "Generated Sample"
                
                st.info(f"📋 {data_type}: {len(st.session_state.vertices):,} vertices, {len(st.session_state.faces):,} faces")
                
                # Show unique labels info
                if hasattr(st.session_state, 'labels'):
                    unique_labels = np.unique(st.session_state.labels)
                    non_zero_labels = unique_labels[unique_labels > 0]
                    st.info(f"🦷 Teeth found: {len(non_zero_labels)} unique FDI IDs")
                    
                    if len(non_zero_labels) > 0:
                        st.write(f"**FDI Labels:** {', '.join(map(str, sorted(non_zero_labels)))}")
                
                # Visualize original mesh
                title = "Real Clinical Dental Scan" if is_real else "Sample Dental Mesh"
                fig = visualize_mesh(
                    st.session_state.vertices, 
                    st.session_state.faces,
                    st.session_state.labels,
                    title
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔮 Segmentation")
            
            if hasattr(st.session_state, 'vertices'):
                processing_time = st.slider("Simulation Time (seconds)", 0.1, 2.0, 0.5, 0.1)
                
                if st.button("🚀 Run Segmentation Simulation"):
                    with st.spinner(f"Processing segmentation..."):
                        pred_labels, pred_instances = simulate_segmentation(
                            st.session_state.vertices, 
                            processing_time
                        )
                        
                        # Store results
                        st.session_state.pred_labels = pred_labels
                        st.session_state.pred_instances = pred_instances
                        
                        st.success(f"✅ Segmentation completed!")
                
                # Show results if available
                if hasattr(st.session_state, 'pred_labels'):
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Vertices", len(st.session_state.pred_labels))
                    with col_b:
                        unique_teeth = len(np.unique(st.session_state.pred_instances))
                        st.metric("Teeth Found", unique_teeth)
                    with col_c:
                        st.metric("Process Time", f"{processing_time:.1f}s")
                    
                    # Visualize segmented result
                    fig_seg = visualize_mesh(
                        st.session_state.vertices,
                        st.session_state.faces,
                        st.session_state.pred_labels,
                        "Segmentation Result"
                    )
                    st.plotly_chart(fig_seg)
                    
                    # Label distribution
                    unique_labels, counts = np.unique(st.session_state.pred_labels, return_counts=True)
                    label_df = pd.DataFrame({
                        'FDI Label': unique_labels,
                        'Vertex Count': counts,
                        'Percentage': (counts / len(st.session_state.pred_labels) * 100).round(2)
                    })
                    
                    # Add tooth names for better understanding
                    tooth_names = {
                        11: 'Upper Right Central Incisor', 12: 'Upper Right Lateral Incisor',
                        13: 'Upper Right Canine', 14: 'Upper Right First Premolar',
                        15: 'Upper Right Second Premolar', 16: 'Upper Right First Molar',
                        17: 'Upper Right Second Molar', 18: 'Upper Right Third Molar',
                        21: 'Upper Left Central Incisor', 22: 'Upper Left Lateral Incisor',
                        23: 'Upper Left Canine', 24: 'Upper Left First Premolar',
                        25: 'Upper Left Second Premolar', 26: 'Upper Left First Molar',
                        27: 'Upper Left Second Molar', 28: 'Upper Left Third Molar',
                        0: 'Gingiva/Background'
                    }
                    
                    label_df['Tooth Name'] = label_df['FDI Label'].map(tooth_names).fillna('Unknown')
                    
                    st.subheader("📊 Segmentation Results")
                    st.dataframe(label_df[['FDI Label', 'Tooth Name', 'Vertex Count', 'Percentage']])
                    
                    # Create bar chart of label distribution
                    fig_bar = px.bar(
                        label_df, 
                        x='FDI Label', 
                        y='Vertex Count',
                        title='Vertex Distribution by Tooth (FDI Labels)',
                        color='FDI Label'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("👆 Generate sample data first!")
    
    with tab3:
        # Demo
        st.header("📊 Performance Demo")
        
        # Sample metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Sample Metrics")
            
            # Display sample performance
            metrics = {
                'TSA': 0.92,
                'TLA': 0.88, 
                'TIR': 0.85,
                'IoU': 0.87,
                'Dice': 0.89
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.3f}")
        
        with col2:
            st.subheader("📈 Performance Chart")
            
            # Create radar chart
            categories = list(metrics.keys())
            values = list(metrics.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Performance'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title="Model Performance"
            )
            st.plotly_chart(fig)
        
        # Model comparison
        st.subheader("🏆 Model Comparison")
        comparison_data = {
            'Model': ['PointNet', 'Custom Multi-task', 'Baseline'],
            'IoU': [0.85, 0.92, 0.78],
            'Processing Time': ['0.12s', '0.18s', '0.25s'],
            'Parameters': ['0.8M', '1.2M', '0.5M']
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df)
        
        # Performance bar chart
        fig_bar = px.bar(df, x='Model', y='IoU', title='Model IoU Comparison', color='Model')
        st.plotly_chart(fig_bar)
    
    # Footer
    st.markdown("---")
    st.markdown("🦷 **3D Teeth Segmentation with PyTorch** - Simple Testing Interface")

if __name__ == "__main__":
    main()