#!/bin/bash
# Frontrun Bot - Setup Script
# Sets up development environment on Linux/Mac/WSL

set -e  # Exit on error

echo "========================================"
echo "  Frontrun Bot - Environment Setup"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ ERROR: Python 3.11+ required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null
echo "✓ pip upgraded"

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install -r requirements.txt
echo "✓ Core dependencies installed"

# Ask about development dependencies
echo ""
read -p "Install development dependencies (pytest, black, mypy)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements-dev.txt
    echo "✓ Development dependencies installed"
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p logs data
echo "✓ Directories created"

# Check config file
echo ""
if [ -f "config/config.yml" ]; then
    echo "✓ Configuration file exists"
else
    echo "⚠ WARNING: config/config.yml not found"
    echo "  Please copy config/config.yml.example and fill in your API keys"
fi

# Run tests to verify setup
echo ""
read -p "Run tests to verify setup? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running unit tests..."
    python run_tests.py unit
fi

echo ""
echo "========================================"
echo "  ✓ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Configure: cp config/config.yml.example config/config.yml"
echo "  3. Edit config.yml with your API keys"
echo "  4. Test RPC: python -m core.rpc_manager"
echo "  5. Run tests: python run_tests.py unit"
echo ""
