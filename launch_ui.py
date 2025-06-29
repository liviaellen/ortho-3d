#!/usr/bin/env python3
"""
Launch script for 3D Teeth Segmentation UI

This script launches the Streamlit web interface for testing the 
PyTorch 3D teeth segmentation system.

Usage:
    python launch_ui.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['streamlit', 'torch', 'trimesh', 'plotly', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def launch_streamlit():
    """Launch the Streamlit app."""
    ui_path = Path(__file__).parent / "ui" / "simple_ui.py"
    
    if not ui_path.exists():
        print(f"❌ UI file not found: {ui_path}")
        return False
    
    print(f"🚀 Launching 3D Teeth Segmentation UI...")
    print(f"📁 UI file: {ui_path}")
    print(f"🌐 The UI will open in your web browser automatically")
    print(f"📝 You can also manually visit: http://localhost:8501")
    print(f"⏹️  Press Ctrl+C to stop the server")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(ui_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.serverAddress", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🦷 3D Teeth Segmentation UI Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if we're in the right directory
    if not Path("models").exists() or not Path("ui").exists():
        print("❌ Please run this script from the project root directory")
        print("   Current directory should contain 'models/' and 'ui/' folders")
        return
    
    # Launch UI
    launch_streamlit()

if __name__ == "__main__":
    main()