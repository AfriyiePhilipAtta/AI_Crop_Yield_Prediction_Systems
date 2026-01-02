#!/bin/bash

# =========================================================
# Full Setup Script for AI Crop Health & Yield Prediction
# =========================================================

# -----------------------------
# Configuration
# -----------------------------
ENV_NAME="crop_env"
PYTHON_VERSION="3.12"

echo "📦 Starting full setup..."

# -----------------------------
# 1️⃣ Check if conda is installed
# -----------------------------
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Install Anaconda or Miniconda first."
    exit 1
fi

# -----------------------------
# 2️⃣ Check if Homebrew is installed (macOS)
# -----------------------------
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Install from https://brew.sh/"
    exit 1
fi

# -----------------------------
# 3️⃣ Initialize conda
# -----------------------------
eval "$(conda shell.bash hook)"

# -----------------------------
# 4️⃣ Create conda environment (if not exists)
# -----------------------------
if conda env list | grep -q "^$ENV_NAME"; then
    echo "✅ Conda environment '$ENV_NAME' already exists"
else
    echo "🐍 Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# -----------------------------
# 5️⃣ Activate environment
# -----------------------------
conda activate "$ENV_NAME"

# -----------------------------
# 6️⃣ Install core geospatial + ML packages
# -----------------------------
echo "🌍 Installing geospatial, ML & satellite packages (conda-forge)..."

conda install -c conda-forge -y \
    geopandas \
    rasterio \
    shapely \
    pyproj \
    fiona \
    gdal \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    xgboost \
    pystac-client \
    planetary-computer

# -----------------------------
# 7️⃣ Install libomp (macOS OpenMP)
# -----------------------------
echo "⚙️ Installing libomp (XGBoost dependency)..."
brew install libomp || true

# -----------------------------
# 8️⃣ Install Earth Engine, geemap, Streamlit
# -----------------------------
echo "🛰 Installing Earth Engine API, geemap & dashboard tools..."

pip install --upgrade pip

pip install \
    earthengine-api \
    geemap \
    streamlit \
    streamlit-folium \
    folium \
    branca \
    tqdm

# -----------------------------
# 9️⃣ Verify installations
# -----------------------------
echo "✅ Verifying installed packages..."

python - << 'EOF'
import importlib.metadata

def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"

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
    import streamlit_folium
    import folium
    import branca
    import tqdm

    print("✅ All packages imported successfully\n")

    print("📦 Package versions:")
    print("-----------------------------")
    print("Python:", version("python"))
    print("GDAL:", gdal.VersionInfo())
    print("GeoPandas:", geopandas.__version__)
    print("Rasterio:", rasterio.__version__)
    print("NumPy:", numpy.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("XGBoost:", xgboost.__version__)
    print("Earth Engine API:", ee.__version__)
    print("Geemap:", geemap.__version__)
    print("Streamlit:", streamlit.__version__)
    print("Streamlit-Folium:", version("streamlit-folium"))
    print("Folium:", folium.__version__)
    print("TQDM:", tqdm.__version__)

except Exception as e:
    print("❌ Verification failed:", e)
    raise
EOF

# -----------------------------
# 🔐 10️⃣ Earth Engine authentication notice
# -----------------------------
echo ""
echo "🔐 FINAL STEP: Authenticate Google Earth Engine"
echo "------------------------------------------------"
echo "Run the following AFTER this script completes:"
echo ""
echo "    conda activate $ENV_NAME"
echo "    earthengine authenticate"
echo ""
echo "This opens a browser window for Google login."
echo "Authentication is required only once per machine."
echo ""
echo "✅ Setup complete!"
