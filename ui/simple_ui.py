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

def visualize_mesh(vertices, faces, labels=None, title="3D Mesh"):
    """Create 3D visualization."""
    if labels is not None:
        # Color mapping for teeth
        colors = [f'hsl({(label-10)*15}, 70%, 50%)' for label in labels]
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
        width=600,
        height=400
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
            
            if st.button("🎲 Generate Sample Jaw"):
                with st.spinner("Creating sample jaw mesh..."):
                    vertices, faces, labels = create_sample_mesh()
                    
                    # Store in session state
                    st.session_state.vertices = vertices
                    st.session_state.faces = faces
                    st.session_state.labels = labels
                    
                    st.success(f"✅ Generated mesh with {len(vertices)} vertices!")
            
            # Show mesh info if generated
            if hasattr(st.session_state, 'vertices'):
                st.info(f"Mesh loaded: {len(st.session_state.vertices)} vertices, {len(st.session_state.faces)} faces")
                
                # Visualize original mesh
                fig = visualize_mesh(
                    st.session_state.vertices, 
                    st.session_state.faces,
                    st.session_state.labels,
                    "Sample Dental Mesh"
                )
                st.plotly_chart(fig)
        
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
                        'Vertex Count': counts
                    })
                    st.dataframe(label_df)
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