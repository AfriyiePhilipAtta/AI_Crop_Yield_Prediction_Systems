#!/usr/bin/env python
# ============================================================
# AI CROP YIELD SYSTEM (OPTIMIZED FOR SMALL DATASETS)
# Uses Sentinel-2 NDVI + Sentinel-1 SAR fusion + XGBoost
# ============================================================

# ============================================================
# --------------------- 1️⃣ IMPORT LIBRARIES ------------------
# ============================================================
import os                   # For managing file paths and directories
import sys                  # For system-level operations, e.g., logging
import logging              # For logging messages during the pipeline
import ee                   # Google Earth Engine Python API
import geemap               # Visualization, exporting maps from GEE
import numpy as np          # Numerical operations on arrays
import pandas as pd         # Dataframe handling for field/ML data
import rasterio             # Read/write GeoTIFF raster files
import matplotlib.pyplot as plt  # For plotting charts
import seaborn as sns       # For enhanced visualization of data

# Machine learning libraries
from xgboost import XGBRegressor                    # Gradient boosting ML model
from sklearn.model_selection import KFold, cross_val_score  # Cross-validation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Model evaluation metrics

# ============================================================
# --------------------- 2️⃣ LOGGING SETUP -------------------
# ============================================================
# Configure logging to track progress and errors
logging.basicConfig(
    level=logging.INFO,  # Only log INFO and higher-level messages
    format="%(asctime)s | %(levelname)s | %(message)s",  # Include timestamp, severity, message
    handlers=[
        logging.StreamHandler(sys.stdout),   # Output logs to console
        logging.FileHandler("pipeline.log")  # Also save logs to a file
    ]
)
logger = logging.getLogger(__name__)  # Create a logger object
logger.info("📌 Logging initialized")  # Confirm logging setup

# ============================================================
# --------------------- 3️⃣ PATHS ---------------------------
# ============================================================
# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      

# Define output directory for generated maps, plots, metadata
OUTPUT_DIR = os.path.join(BASE_DIR, "output")             
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create folder if it does not exist

# Input CSV with field observations: NDVI, VV, VH, rainfall, temperature, yield
FIELD_CSV = os.path.join(BASE_DIR, "DS_5_Fielddata.csv")  

# Paths for output raster files
NDVI_MAP_PATH = os.path.join(OUTPUT_DIR, "NDVI_Map.tif")  
YIELD_MAP_PATH = os.path.join(OUTPUT_DIR, "Yield_Map.tif")  
UNCERTAINTY_MAP_PATH = os.path.join(OUTPUT_DIR, "Yield_Uncertainty_Map.tif")  

# ============================================================
# --------------------- 4️⃣ EARTH ENGINE INIT ----------------
# ============================================================
PROJECT_ID = "quiet-subset-447718-q0"  # Your Google Earth Engine project ID

# Initialize Earth Engine API; authenticate if needed
try:
    ee.Initialize(project=PROJECT_ID)  # Try initializing with credentials
except Exception:
    ee.Authenticate()                  # Authenticate via browser if needed
    ee.Initialize(project=PROJECT_ID)  # Retry initialization

logger.info("🌍 Google Earth Engine initialized")  # Log successful init

# ============================================================
# --------------------- 5️⃣ AREA OF INTEREST -----------------
# ============================================================
# Define farm area polygon using coordinates (longitude, latitude)
AOI = ee.Geometry.Polygon([
    [
        [10.62976, 52.32970],
        [10.64041, 52.32966],
        [10.64071, 52.33370],
        [10.62963, 52.33378],
        [10.62976, 52.32970]
    ]
])
# Add a buffer around AOI to avoid edge artifacts during raster operations
AOI_BUF = AOI.buffer(75)
logger.info("📍 AOI defined with buffer")  

# ============================================================
# --------------------- 6️⃣ SENTINEL-2 COLLECTION ----------
# ============================================================
# Load Sentinel-2 surface reflectance images that overlap AOI
# and satisfy date and cloud coverage filters
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI_BUF)                    # Only images overlapping AOI
    .filterDate("2025-01-01", "2025-12-30")  # Year-long collection
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # Less than 20% clouds
)
logger.info(f"🛰️ Sentinel-2 images found: {s2.size().getInfo()}")  

# ============================================================
# --------------------- 7️⃣ SENTINEL-1 COLLECTION ----------
# ============================================================
# Load Sentinel-1 SAR images with VV and VH polarizations
# Filter by AOI, date, orbit, and instrument mode
s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(AOI_BUF)
    .filterDate("2025-01-01", "2025-12-30")
    .filter(ee.Filter.eq("instrumentMode", "IW"))  # Interferometric Wide mode
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    .select(["VV", "VH"])  # Only select VV and VH bands
)
logger.info(f"🛰️ Sentinel-1 images found: {s1.size().getInfo()}")  

# ============================================================
# --------------------- 8️⃣ NDVI CALCULATION ----------------
# ============================================================
# Function to compute NDVI and mask non-vegetation pixels
def add_ndvi(img):
    """
    Compute NDVI = (NIR - RED) / (NIR + RED)
    Keep only vegetation pixels using Scene Classification Layer (SCL)
    """
    ndvi = img.normalizedDifference(["B8","B4"]).rename("NDVI")  # B8=NIR, B4=Red
    scl = img.select("SCL")  # Scene Classification Layer
    # Keep vegetation: 4=crop, 5=tree, 6=shrub
    valid = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    ndvi = ndvi.updateMask(valid)  # Mask out non-vegetation
    return img.addBands(ndvi)      # Add NDVI as new band

s2_ndvi = s2.map(add_ndvi)  # Apply NDVI function to all Sentinel-2 images
logger.info("✅ NDVI computed for all Sentinel-2 images")  

# ============================================================
# --------------------- 9️⃣ SAR VEGETATION INDEX -----------
# ============================================================
# Function to compute SAR vegetation index (VH/VV)
def add_s1_index(img):
    """
    Compute SAR vegetation index: VH / (VV + 1e-6)
    Small constant avoids division by zero
    """
    s1_vi = img.expression("VH / (VV + 1e-6)", {"VV": img.select("VV"), "VH": img.select("VH")}).rename("S1_VI")
    return img.addBands(s1_vi)

s1_vi = s1.map(add_s1_index)  # Apply SAR vegetation index to all images
logger.info("✅ SAR vegetation index computed for Sentinel-1")  

# ============================================================
# --------------------- 🔟 CLOUD-SAFE NDVI -------------------
# ============================================================
# Create median composites to reduce cloud and noise
ndvi_s2 = s2_ndvi.select("NDVI").median()  # Median NDVI from S2
s1_composite = s1_vi.select("S1_VI").median()  # Median SAR VI
s1_scaled = s1_composite.unitScale(0,1)  # Scale SAR index between 0 and 1
ndvi_fused = ndvi_s2.unmask(s1_scaled)   # Replace missing NDVI with SAR proxy
ndvi_composite = ndvi_fused.clip(AOI_BUF)  # Clip final NDVI to AOI
logger.info("☁️ Cloud-safe NDVI composite created")  

# ============================================================
# --------------------- 1️⃣1️⃣ FIELD NDVI STATS -------------
# ============================================================
# Compute field-level NDVI statistics (mean & max) for ML features
stats = ndvi_composite.reduceRegion(
    reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.max(), sharedInputs=True),
    geometry=AOI_BUF,
    scale=10,
    maxPixels=1e9
)
mean_ndvi = stats.get("NDVI_mean").getInfo()
max_ndvi = stats.get("NDVI_max").getInfo()
logger.info(f"📊 Field NDVI stats -> Mean: {mean_ndvi:.3f}, Max: {max_ndvi:.3f}")  

# ============================================================
# --------------------- 1️⃣2️⃣ EXPORT NDVI MAP --------------
# ============================================================
# Export NDVI composite to GeoTIFF
geemap.ee_export_image(ndvi_composite, filename=NDVI_MAP_PATH, scale=10, region=AOI_BUF)
logger.info(f"🗺️ NDVI map exported: {NDVI_MAP_PATH}")  

# ============================================================
# --------------------- 1️⃣3️⃣ LOAD FIELD DATA --------------
# ============================================================
# Load field observations CSV for machine learning
field_data = pd.read_csv(FIELD_CSV)
FEATURES = ["NDVI_mean","NDVI_max","VV","VH","rainfall","temp"]  # Predictor variables
X = field_data[FEATURES]
y = field_data["yield"]  # Target variable
logger.info(f"📂 Field data loaded: {len(X)} observations")  

# ============================================================
# --------------------- 1️⃣4️⃣ MACHINE LEARNING ------------
# ============================================================
# Initialize XGBoost regressor (suitable for small datasets)
model = XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    min_child_weight=3, gamma=0.1, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1
)

# 5-fold cross-validation for model evaluation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
cv_mae = -cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv)
cv_r2 = cross_val_score(model, X, y, scoring="r2", cv=cv)

yield_uncertainty = cv_rmse.mean()  # Use CV RMSE as uncertainty estimate
model.fit(X, y)  # Train XGBoost on full dataset
logger.info("✅ XGBoost model trained on full dataset")  

# ============================================================
# --------------------- 1️⃣5️⃣ FEATURE IMPORTANCE ----------
# ============================================================
# Get importance of each feature
feature_importance = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
# Plot feature importance
plt.figure(figsize=(8,5))
sns.barplot(data=feature_importance, x="importance", y="feature", palette="viridis")
plt.title("Feature Importance for Yield Prediction")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=300)
plt.close()
logger.info("📈 Feature importance saved")  

# ============================================================
# --------------------- 1️⃣6️⃣ TRAINING METRICS -------------
# ============================================================
# Evaluate model on training data
y_pred_train = model.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
train_r2 = r2_score(y, y_pred_train)
train_mae = mean_absolute_error(y, y_pred_train)
logger.info(f"Training RMSE: {train_rmse:.3f}, R²: {train_r2:.3f}")  

# ============================================================
# --------------------- 1️⃣7️⃣ FIELD-LEVEL YIELD ------------
# ============================================================
# Predict field-level yield using average stats
X_field = [[mean_ndvi, max_ndvi, field_data["VV"].mean(), field_data["VH"].mean(),
            field_data["rainfall"].mean(), field_data["temp"].mean()]]
predicted_yield = model.predict(X_field)[0]
logger.info(f"🌾 Predicted yield: {predicted_yield:.2f} ± {yield_uncertainty:.2f} t/ha")  

# ============================================================
# --------------------- 1️⃣8️⃣ YIELD & UNCERTAINTY MAP -------
# ============================================================
# Generate yield raster by scaling NDVI with predicted yield
with rasterio.open(NDVI_MAP_PATH) as src:
    ndvi = src.read(1)
    meta = src.meta.copy()
ndvi = np.where(ndvi <= 0, np.nan, ndvi)  # Ignore negative/no-data pixels
yield_map = ndvi * predicted_yield / np.nanmean(ndvi)  # Scale to yield
uncertainty_map = yield_map * (yield_uncertainty / predicted_yield)  # Uncertainty map
meta.update(dtype="float32", count=1, nodata=np.nan)

# Save maps
with rasterio.open(YIELD_MAP_PATH, "w", **meta) as dst:
    dst.write(yield_map.astype("float32"), 1)
with rasterio.open(UNCERTAINTY_MAP_PATH, "w", **meta) as dst:
    dst.write(uncertainty_map.astype("float32"), 1)
logger.info(f"🗺️ Yield & uncertainty maps exported")  

# ============================================================
# --------------------- 1️⃣9️⃣ SAVE MODEL METADATA ----------
# ============================================================
# Save all relevant model metadata for future reference
metadata = {
    "model": "XGBoost Regressor (Optimized for Small Data)",
    "n_observations": len(X),
    "features": FEATURES,
    "cv_rmse": float(cv_rmse.mean()),
    "cv_mae": float(cv_mae.mean()),
    "cv_r2": float(cv_r2.mean()),
    "train_rmse": float(train_rmse),
    "train_r2": float(train_r2),
    "predicted_yield": float(predicted_yield),
    "uncertainty": float(yield_uncertainty),
    "feature_importance": feature_importance.to_dict("records")
}

metadata_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
import json
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
logger.info(f"🗂️ Model metadata saved: {metadata_path}")  

# ============================================================
# --------------------- 2️⃣0️⃣ PIPELINE COMPLETE ------------
# ============================================================
logger.info("✅ AI CROP YIELD PIPELINE COMPLETE")  