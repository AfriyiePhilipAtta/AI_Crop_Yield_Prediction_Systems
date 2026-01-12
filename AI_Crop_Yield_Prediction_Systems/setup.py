import subprocess
import sys

# -----------------------------
# USER-DEFINED CONFIGURATION
# -----------------------------
ENV_NAME = "crop_env"         # Name of the conda environment to create
PYTHON_VERSION = "3.12"       # Python version to use in the environment

print("📦 Starting Windows full setup...")

# -----------------------------
# 1️⃣ Create conda environment
# -----------------------------
try:
    subprocess.run(
        ["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"],
        check=True
    )
    # ✅ Explanation:
    # 'conda create -n <ENV_NAME> python=<version> -y' creates a new isolated environment.
    # This environment keeps all packages and dependencies separate from your system Python,
    # preventing version conflicts and ensuring reproducibility.
    # '-y' automatically confirms installation.
    print(f"✅ Conda environment '{ENV_NAME}' created.")
except subprocess.CalledProcessError:
    # If the environment already exists, this exception will occur
    print(f"⚠️ Environment '{ENV_NAME}' may already exist.")

# -----------------------------
# 2️⃣ Activate environment and install packages
# -----------------------------
packages = [
    "geopandas",            # Handles geospatial vector data (shapefiles, GeoJSON)
    "rasterio",             # Reads/writes raster data (e.g., GeoTIFF)
    "shapely",              # Geometry operations (points, lines, polygons)
    "pyproj",               # Coordinate reference system transformations
    "fiona",                # I/O for vector data formats
    "gdal",                 # Geospatial Data Abstraction Library (core GIS engine)
    "numpy",                # Numerical operations on arrays
    "pandas",               # Tabular data manipulation
    "matplotlib",           # 2D plotting library
    "seaborn",              # Statistical plotting (enhances matplotlib)
    "scikit-learn",         # Machine learning algorithms (XGBoost requires it)
    "xgboost",              # Gradient boosting library for regression/classification
    "pystac-client",        # Access to STAC catalogs (spatial-temporal asset catalogs)
    "planetary-computer",   # Tools for Microsoft Planetary Computer datasets
    "earthengine-api",      # Google Earth Engine Python API
    "geemap",               # Interactive mapping & exporting for Earth Engine
    "streamlit",            # Web app framework for Python
    "streamlit-folium",     # Embeds interactive folium maps in Streamlit apps
    "folium",               # Leaflet.js maps in Python
    "branca",               # Helper library for folium colormaps and legends
    "tqdm"                  # Progress bars for loops/downloads
]

# Explanation:
# This list contains all the libraries needed for geospatial data processing,
# visualization, machine learning, and interactive dashboards.
# Installing everything in one go ensures reproducibility.

print("🌍 Installing packages...")

subprocess.run(
    ["conda", "install", "-n", ENV_NAME, "-c", "conda-forge", "-y"] + packages,
    check=True
)
# ✅ Explanation:
# 'conda install -n <ENV_NAME> -c conda-forge <packages>'
# Installs all packages in the previously created environment.
# '-c conda-forge' specifies the conda-forge channel (community-maintained, often newest versions).
# 'check=True' ensures the script stops if installation fails.

print("\n✅ All packages installation attempted.")

# -----------------------------
# 3️⃣ Google Earth Engine Authentication Instructions
# -----------------------------
print("\n🔐 FINAL STEP: Authenticate Google Earth Engine")
print("Run the following after this setup completes:")
print(f"    conda activate {ENV_NAME}")
print("    earthengine authenticate")
# Explanation:
# - 'conda activate <ENV_NAME>': switches to the new environment.
# - 'earthengine authenticate': opens a browser to log in with your Google account
#   and grants access to Earth Engine datasets. This step is required only once per machine.
