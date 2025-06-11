#!/bin/bash
# Script to set up the Detectron2 training environment

echo "===== Detectron2 Training Environment Setup ====="
echo "This script will set up the necessary environment for the Detectron2 project."

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed. Please check the error messages above."
        exit 1
    fi
}

# Check if Python is installed
python --version
check_status "Python check"
echo "✓ Python is installed"

# Create virtual environment if it doesn't exist
if [ ! -d "detectron2-env" ]; then
    echo "Creating virtual environment..."
    python -m venv detectron2-env
    check_status "Virtual environment creation"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source detectron2-env/bin/activate
check_status "Virtual environment activation"
echo "✓ Virtual environment activated"

# Update pip
echo "Updating pip..."
pip install --upgrade pip
check_status "Pip update"
echo "✓ Pip updated"

# Install dependencies
echo "Installing required packages (this may take a while)..."
pip install -r requirements.txt
check_status "Package installation"
echo "✓ Packages installed"

# Check if CUDA is available
echo "Checking for CUDA availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo "===== Setup Complete ====="
echo "To activate the environment in the future, run:"
echo "source detectron2-env/bin/activate"
echo ""
echo "To start training with minimal configuration:"
echo "python train.py --data-dir ./data --class-list ./class.names"
echo ""
echo "For more options, see README.md or run:"
echo "python train.py --help"
