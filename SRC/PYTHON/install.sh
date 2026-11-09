#!/bin/bash

set -e  # Exit on error
set -o pipefail

# Create virtual environment in ./env if it doesn't exist
if [ ! -d "env" ]; then
    echo "🛠 Creating virtual environment in $(pwd)/.env ..."
    python3 -m venv .env --prompt TeNet
else
    echo "✅ Virtual environment already exists at $(pwd)/.env"
fi

# Activate the virtual environment
echo "📦 Activating virtual environment ..."
source .env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip ..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📄 Installing dependencies from requirements.txt ..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found — skipping dependency install."
fi

# Cythonize Cython extensions
echo "🛠 Cythonize the relevant scripts ..."
cd tenet/algorithms/core
python setup.py build_ext --inplace
cd ../../..

# Install project in editable mode
echo "🔗 Installing package in editable mode ..."
mkdir -p .build
pip install -e .

echo "✅ Environment setup complete. To activate run"
echo "   source .env/bin/activate"