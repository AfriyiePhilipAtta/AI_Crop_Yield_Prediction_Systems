#!/bin/bash

# =========================================================
# Full Setup Script for AI Crop Health & Yield Prediction
# =========================================================

# Name of the conda environment to create
ENV_NAME="crop_env"
# Python version to use
PYTHON_VERSION="3.12"

echo "📦 Starting full setup..."

# ---------------------------------------------------------
# 1️⃣ Check if conda is installed
# ---------------------------------------------------------
# Ensures that Anaconda or Miniconda is installed before continuing
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# ---------------------------------------------------------
# 2️⃣ Check if Homebrew is installed (macOS only)
# ---------------------------------------------------------
# Homebrew is required for installing some dependencies (e.g., libomp)
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
    exit 1
fi

# ---------------------------------------------------------
# 3️⃣ Initialize conda
# ---------------------------------------------------------
# Sets up conda in the current shell session
eval "$(conda shell.bash hook)"

# ---------------------------------------------------------
# 4️⃣ Create conda environment
# ---------------------------------------------------------
echo "🐍 Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
# Creates a new isolated environment with the specified Python version
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# ---------------------------------------------------------
# 5️⃣ Activate environment
# ---------------------------------------------------------
# Activates the newly created environment so all subsequent installs go there
conda activate $ENV_NAME

# ---------------------------------------------------------
# 6️⃣ Install core geospatial + ML packages via conda-forge
# ---------------------------------------------------------
echo "🌍 Installing geospatial + ML + satellite packages..."
# Install packages for:
# - Geospatial processing: geopandas, rasterio, shapely, pyproj, fiona, gdal
# - Numerical & ML: numpy, pandas, matplotlib, scikit-learn, xgboost
# - Satellite data handling: pystac-client, planetary-computer
conda install -c conda-forge \
    geopandas rasterio shapely pyproj fiona gdal \
    numpy pandas matplotlib scikit-learn \
    xgboost \
    pystac-client planetary-computer \
    -y

# ---------------------------------------------------------
# 7️⃣ Install libomp for XGBoost (macOS requirement)
# ---------------------------------------------------------
echo "⚙️ Installing libomp (required for XGBoost on macOS)..."
# OpenMP library is needed for XGBoost parallel computation on macOS
brew install libomp || true   # Ignore errors if already installed

# ---------------------------------------------------------
# 8️⃣ Install Earth Engine, geemap, Streamlit & dashboard deps
# ---------------------------------------------------------
echo "🛰 Installing Earth Engine API, geemap, Streamlit & dashboard packages..."
pip install --upgrade pip
# pip install packages for:
# - Google Earth Engine: earthengine-api
# - Python interface & visualization for GEE: geemap
# - Web dashboard: streamlit, streamlit-folium, folium, branca
# - Progress bars: tqdm
pip install \
    earthengine-api \
    geemap \
    streamlit \
    streamlit-folium \
    folium \
    branca \
    tqdm

# ---------------------------------------------------------
# 9️⃣ Verify installations
# ---------------------------------------------------------
echo "✅ Verifying installed packages..."
# Use an inline Python script to test that all packages can be imported
python - << 'EOF'
try:
    import rasterio
    import geopandas
    import shapely
    import pyproj
    import fiona
    from osgeo import gdal
    import numpy
    import pandas
    import matplotlib
    import sklearn
    import xgboost
    import pystac_client
    import planetary_computer
    import ee
    import geemap
    import streamlit
    import tqdm
    import streamlit_folium
    import folium
    import branca

    print("✅ All packages installed successfully")
    print("GDAL version:", gdal.VersionInfo())
    print("Earth Engine API version:", ee.__version__)
    print("Streamlit version:", streamlit.__version__)
    print("streamlit-folium version:", streamlit_folium.__version__)
except Exception as e:
    print("❌ Package installation failed:", e)
    raise
EOF

# ---------------------------------------------------------
# 🔐 10️⃣ Earth Engine Authentication (USER ACTION REQUIRED)
# ---------------------------------------------------------
echo ""
echo "🔐 FINAL STEP: Authenticate Google Earth Engine"
echo "------------------------------------------------"
echo "Run the following command AFTER this script finishes:"
echo ""
echo "    conda activate $ENV_NAME"
echo "    earthengine authenticate"
echo ""
echo "This will open a browser window for Google login."
echo "Authentication is required ONLY ONCE per machine."
echo ""
echo "✅ Setup complete!"
