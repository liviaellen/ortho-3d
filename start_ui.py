#!/usr/bin/env python3
"""
Simple UI Launcher for 3D Teeth Segmentation

Just launches the working simple UI without complex checks.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🦷 Starting 3D Teeth Segmentation UI...")
    print("🌐 Opening at: http://localhost:8503")
    print("⏹️  Press Ctrl+C to stop")
    
    ui_path = Path(__file__).parent / "ui" / "simple_ui.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(ui_path),
            "--server.port", "8503"
        ])
    except KeyboardInterrupt:
        print("\n🛑 UI stopped")

if __name__ == "__main__":
    main()