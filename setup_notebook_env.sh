#!/bin/bash

echo "🦷 Setting up environment for 3D Teeth Segmentation Notebook"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "3D_Teeth_Segmentation_Final_Project.ipynb" ]; then
    echo "❌ Please run this script from the project directory"
    exit 1
fi

# Option 1: Try ortho environment
echo "🔍 Checking ortho environment..."
if pyenv activate ortho 2>/dev/null; then
    echo "✅ Using ortho environment"
    ENV_NAME="ortho"
else
    echo "⚠️  ortho environment not available, using current ds24"
    ENV_NAME="ds24"
fi

echo "📦 Installing required packages..."

# Install core packages
pip install torch torchvision torchaudio --quiet
pip install trimesh --quiet
pip install plotly --quiet
pip install jupyter --quiet
pip install matplotlib --quiet
pip install seaborn --quiet
pip install pandas --quiet
pip install numpy --quiet
pip install scikit-learn --quiet
pip install tqdm --quiet

echo "✅ Environment setup complete!"
echo ""
echo "🚀 To start the notebook:"
echo "   jupyter notebook 3D_Teeth_Segmentation_Final_Project.ipynb"
echo ""
echo "🌐 Or use Jupyter Lab:"
echo "   jupyter lab"
echo ""
echo "Current environment: $ENV_NAME"
echo "Python version: $(python --version)"