#!/usr/bin/env python
# ============================================================
# AI CROP YIELD SYSTEM (OPTIMIZED FOR SMALL DATASETS)
# Sentinel‑2 NDVI + Sentinel‑1 SAR FUSION + XGBoost
# ============================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import os                   # File paths, directories
import sys                  # System-level operations
import logging              # Logging pipeline steps
import ee                   # Google Earth Engine API
import geemap               # Visualization & export for GEE
import numpy as np          # Array/numerical computations
import pandas as pd         # Tabular data handling
import rasterio             # Reading/writing GeoTIFF raster files

from xgboost import XGBRegressor                    # ML model
from sklearn.model_selection import KFold, cross_val_score  # Cross-validation
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------------------
# LOGGING SETUP
# -----------------------------
# Logs messages to both console and a file "pipeline.log."
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console
        logging.FileHandler("pipeline.log") # File
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# PATHS
# -----------------------------
# Base directory is where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output folder if missing

# Input field-level CSV (yield, NDVI, SAR, weather)
FIELD_CSV = os.path.join(BASE_DIR, "DS_5_Fielddata.csv")

# Output GeoTIFF paths
NDVI_MAP_PATH = os.path.join(OUTPUT_DIR, "NDVI_Map.tif")
YIELD_MAP_PATH = os.path.join(OUTPUT_DIR, "Yield_Map.tif")
UNCERTAINTY_MAP_PATH = os.path.join(OUTPUT_DIR, "Yield_Uncertainty_Map.tif")

# -----------------------------
# EARTH ENGINE INITIALIZATION
# -----------------------------
PROJECT_ID = "quiet-subset-447718-q0"
try:
    ee.Initialize(project=PROJECT_ID)  # Try initializing GEE
except Exception:
    ee.Authenticate()                 # Authenticate if not logged in
    ee.Initialize(project=PROJECT_ID)

logger.info("Earth Engine initialized")

# -----------------------------
# AREA OF INTEREST (AOI)
# -----------------------------
# Define field/farm polygon using lat/lon coordinates
AOI = ee.Geometry.Polygon([
    [
        [10.62976, 52.32970],
        [10.64041, 52.32966],
        [10.64071, 52.33370],
        [10.62963, 52.33378],
        [10.62976, 52.32970]
    ]
])
# Add 75m buffer to avoid edge effects
AOI_BUF = AOI.buffer(75)

# -----------------------------
# SENTINEL-2 IMAGE COLLECTION
# -----------------------------
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI_BUF)                    # Only images overlapping AOI
    .filterDate("2025-01-01", "2025-12-30")  # Time period
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # Filter clouds
)
logger.info(f"Sentinel‑2 scenes found: {s2.size().getInfo()}")

# -----------------------------
# SENTINEL-1 SAR COLLECTION (CLOUD-RESILIENT)
# -----------------------------
s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(AOI_BUF)
    .filterDate("2025-01-01", "2025-12-30")
    .filter(ee.Filter.eq("instrumentMode", "IW"))  # Interferometric Wide swath
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))  # VV polarization
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))  # VH polarization
    .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    .select(["VV", "VH"])  # Keep only VV and VH bands
)
logger.info(f"Sentinel‑1 scenes found: {s1.size().getInfo()}")

# -----------------------------
# NDVI CALCULATION FOR SENTINEL-2
# -----------------------------
def add_ndvi(img):
    """
    Compute NDVI for each Sentinel-2 image and mask non-vegetation pixels.
    NDVI = (NIR - RED) / (NIR + RED)
    """
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")  # B8=NIR, B4=RED
    scl = img.select("SCL")  # Scene classification (cloud, vegetation, water, etc.)
    
    # Keep only vegetation classes 4,5,6 (green, crop, tree)
    valid = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    ndvi = ndvi.updateMask(valid)  # Mask out everything else
    
    return img.addBands(ndvi)

s2_ndvi = s2.map(add_ndvi)  # Apply NDVI calculation to all images

# -----------------------------
# SENTINEL-1 VEGETATION INDEX
# -----------------------------
def add_s1_index(img):
    """
    Create a SAR-based vegetation index: VH / (VV + 1e-6)
    Small epsilon avoids division by zero
    """
    s1_vi = img.expression(
        "VH / (VV + 1e-6)",
        {
            "VV": img.select("VV"),
            "VH": img.select("VH")
        }
    ).rename("S1_VI")
    return img.addBands(s1_vi)

s1_vi = s1.map(add_s1_index)  # Apply SAR vegetation index

# -----------------------------
# CLOUD-SAFE NDVI COMPOSITE
# -----------------------------
# Compute median over time to reduce noise
ndvi_s2 = s2_ndvi.select("NDVI").median()
s1_composite = s1_vi.select("S1_VI").median()

# Scale SAR index to 0–1
s1_scaled = s1_composite.unitScale(0, 1)
# Fill missing NDVI pixels with SAR vegetation proxy
ndvi_fused = ndvi_s2.unmask(s1_scaled)
# Clip to AOI
ndvi_composite = ndvi_fused.clip(AOI_BUF)
logger.info("Cloud‑safe NDVI composite created")

# -----------------------------
# FIELD-LEVEL NDVI STATISTICS
# -----------------------------
stats = ndvi_composite.reduceRegion(
    reducer=ee.Reducer.mean().combine(
        reducer2=ee.Reducer.max(),  # Compute mean and max NDVI
        sharedInputs=True
    ),
    geometry=AOI_BUF,
    scale=10,          # 10m resolution
    maxPixels=1e9      # Allow large areas
)

mean_ndvi = stats.get("NDVI_mean").getInfo()
max_ndvi = stats.get("NDVI_max").getInfo()
logger.info(f"Mean NDVI: {mean_ndvi:.3f}")
logger.info(f"Max NDVI:  {max_ndvi:.3f}")

# -----------------------------
# EXPORT NDVI MAP
# -----------------------------
geemap.ee_export_image(
    ndvi_composite,
    filename=NDVI_MAP_PATH,
    scale=10,
    region=AOI_BUF,
    file_per_band=False
)
logger.info("NDVI composite exported")

# -----------------------------
# LOAD FIELD DATA
# -----------------------------
field_data = pd.read_csv(FIELD_CSV)

FEATURES = [
    "NDVI_mean",
    "NDVI_max",
    "VV",
    "VH",
    "rainfall",
    "temp"
]

X = field_data[FEATURES]  # Predictors
y = field_data["yield"]    # Target variable

logger.info(f"Dataset size: {len(X)} observations")
logger.info(f"Features: {FEATURES}")

# -----------------------------
# MACHINE LEARNING MODEL (XGBoost)
# -----------------------------
# Optimized for small datasets with strong regularization
model = XGBRegressor(
    n_estimators=100,        # Reduced from 300 to prevent overfitting
    max_depth=3,             # Shallower trees for small data
    learning_rate=0.05,      # Conservative learning rate
    min_child_weight=3,      # Requires more samples per leaf
    gamma=0.1,               # Minimum loss reduction for splits
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Column sampling
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    random_state=42,
    n_jobs=-1
)

# 5-fold cross-validation to estimate performance and uncertainty
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Calculate multiple metrics via cross-validation
cv_rmse = -cross_val_score(
    model, X, y,
    scoring="neg_root_mean_squared_error",
    cv=cv
)

cv_mae = -cross_val_score(
    model, X, y,
    scoring="neg_mean_absolute_error",
    cv=cv
)

cv_r2 = cross_val_score(
    model, X, y,
    scoring="r2",
    cv=cv
)

# Report cross-validation results
logger.info("=" * 50)
logger.info("CROSS-VALIDATION RESULTS (5-Fold)")
logger.info("=" * 50)
logger.info(f"RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
logger.info(f"MAE:  {cv_mae.mean():.3f} ± {cv_mae.std():.3f}")
logger.info(f"R²:   {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
logger.info("=" * 50)

yield_uncertainty = cv_rmse.mean()  # Average RMSE across folds

# Train final model on all data
model.fit(X, y)
logger.info("Model trained on full dataset")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

logger.info("\nFeature Importance:")
for idx, row in feature_importance.iterrows():
    logger.info(f"  {row['feature']}: {row['importance']:.4f}")

# Calculate training metrics
y_pred_train = model.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))
train_r2 = r2_score(y, y_pred_train)
train_mae = mean_absolute_error(y, y_pred_train)

logger.info("\nTraining Set Performance:")
logger.info(f"  RMSE: {train_rmse:.3f}")
logger.info(f"  MAE:  {train_mae:.3f}")
logger.info(f"  R²:   {train_r2:.3f}")

# -----------------------------
# FIELD-LEVEL YIELD PREDICTION
# -----------------------------
X_field = [[
    mean_ndvi,
    max_ndvi,
    field_data["VV"].mean(),
    field_data["VH"].mean(),
    field_data["rainfall"].mean(),
    field_data["temp"].mean()
]]

predicted_yield = model.predict(X_field)[0]
logger.info("\n" + "=" * 50)
logger.info(f"PREDICTED YIELD: {predicted_yield:.2f} ± {yield_uncertainty:.2f} t/ha")
logger.info("=" * 50)

# -----------------------------
# YIELD MAP & UNCERTAINTY MAP
# -----------------------------
with rasterio.open(NDVI_MAP_PATH) as src:
    ndvi = src.read(1)
    meta = src.meta.copy()

# Mask invalid NDVI values
ndvi = np.where(ndvi <= 0, np.nan, ndvi)

# Scale NDVI to yield map
yield_map = ndvi * predicted_yield / np.nanmean(ndvi)
# Calculate per-pixel uncertainty
uncertainty_map = yield_map * (yield_uncertainty / predicted_yield)

# Update metadata for float32 and nodata
meta.update(dtype="float32", count=1, nodata=np.nan)

# Export yield map
with rasterio.open(YIELD_MAP_PATH, "w", **meta) as dst:
    dst.write(yield_map.astype("float32"), 1)

# Export uncertainty map
with rasterio.open(UNCERTAINTY_MAP_PATH, "w", **meta) as dst:
    dst.write(uncertainty_map.astype("float32"), 1)

logger.info("\nYield map exported: " + YIELD_MAP_PATH)
logger.info("Uncertainty map exported: " + UNCERTAINTY_MAP_PATH)

# -----------------------------
# SAVE MODEL METADATA
# -----------------------------
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
    "feature_importance": feature_importance.to_dict('records')
}

import json
metadata_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

logger.info(f"Model metadata saved: {metadata_path}")

# -----------------------------
# PIPELINE COMPLETE
# -----------------------------
logger.info("\n" + "=" * 50)
logger.info("✅ AI CROP YIELD PIPELINE COMPLETE")
logger.info("=" * 50)
