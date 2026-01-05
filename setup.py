import subprocess
import sys

ENV_NAME = "crop_env"
PYTHON_VERSION = "3.12"

print("📦 Starting Windows full setup...")

# -----------------------------
# 1️⃣ Create conda environment
# -----------------------------
try:
    subprocess.run(["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"], check=True)
    print(f"✅ Conda environment '{ENV_NAME}' created.")
except subprocess.CalledProcessError:
    print(f"⚠️ Environment '{ENV_NAME}' may already exist.")

# -----------------------------
# 2️⃣ Activate environment and install packages
# -----------------------------
packages = [
    "geopandas",
    "rasterio",
    "shapely",
    "pyproj",
    "fiona",
    "gdal",
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "xgboost",
    "pystac-client",
    "planetary-computer",
    "earthengine-api",
    "geemap",
    "streamlit",
    "streamlit-folium",
    "folium",
    "branca",
    "tqdm"
]

# Install packages in the environment
print("🌍 Installing packages...")
subprocess.run(["conda", "install", "-n", ENV_NAME, "-c", "conda-forge", "-y"] + packages, check=True)

print("\n✅ All packages installation attempted.")

print("\n🔐 FINAL STEP: Authenticate Google Earth Engine")
print("Run the following after this setup completes:")
print(f"    conda activate {ENV_NAME}")
print("    earthengine authenticate")
print("This opens a browser for Google login. Authentication is needed only once per machine.")
