#!/bin/bash
# VM Setup Script for In-Vehicle AI Copilot
# Run this on the GCP VM after cloning the repo

set -e

echo "========================================"
echo "🚗 In-Vehicle AI Copilot - VM Setup"
echo "========================================"

# Check if we're on the VM
if ! nvidia-smi &> /dev/null; then
    echo "❌ This script should be run on the GPU VM"
    exit 1
fi

echo "✅ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Install UV if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
fi

echo "📦 UV version: $(uv --version)"

# Create virtual environment with Python 3.10
echo "🐍 Setting up Python environment..."
uv venv --python 3.10

# Activate venv
source .venv/bin/activate

# Install base dependencies
echo "📦 Installing base dependencies..."
uv pip install kaggle pandas numpy pillow tqdm pyyaml scikit-learn matplotlib seaborn

# Install ML dependencies (works on Linux with GPU)
echo "📦 Installing ML dependencies..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers peft accelerate bitsandbytes datasets

# Verify installations
echo ""
echo "========================================"
echo "📊 Installation Verification"
echo "========================================"

python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import transformers
print(f"Transformers: {transformers.__version__}")

import peft
print(f"PEFT: {peft.__version__}")
EOF

# Setup Kaggle credentials if not present
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "⚠️  Kaggle credentials not found."
    echo "   Please copy your kaggle.json to ~/.kaggle/"
    echo "   Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
fi

echo ""
echo "========================================"
echo "✅ VM Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate venv: source .venv/bin/activate"
echo "  2. Download data: python scripts/download_data.py --model person_b"
echo "  3. Prepare data:"
echo "     python models/drowsiness_detector/prepare_data.py"
echo "     python models/distraction_detector/prepare_data.py"
echo "  4. Train models:"
echo "     python models/drowsiness_detector/train.py --baseline"
echo "     python models/distraction_detector/train.py --baseline"
